#!/usr/bin/env python3
"""Dependency-free reference for the *active main.cpp* feature pipeline.

It accepts only 16 kHz, mono, signed-PCM16 WAV or a 16,000-sample raw PCM16LE
file. It intentionally rejects resampling/downmixing so no undocumented audio
conversion changes the model input. Outputs use TFLite flat NHWC order.
"""
import argparse, json, math, re, struct, wave
from pathlib import Path

SR, SAMPLES, NFFT, HOP, FRAMES, MELS, MFCCS = 16000, 16000, 480, 320, 49, 40, 40
MEAN, STD, SCALE, ZP = -5.2158537, 46.219826, 0.0480119027197361, 49

def read_pcm(path):
    p = Path(path)
    if p.suffix.lower() == '.wav':
        with wave.open(str(p), 'rb') as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (SR, 1, 2, 'NONE'):
                raise ValueError('WAV must be 16 kHz, mono, PCM16; resampling/downmixing is intentionally not implicit')
            raw = w.readframes(w.getnframes())
    else: raw = p.read_bytes()
    if len(raw) != SAMPLES * 2:
        raise ValueError(f'input must contain exactly {SAMPLES} PCM16 samples; got {len(raw)//2}')
    return struct.unpack('<%dh' % SAMPLES, raw)

def read_c_header(path):
    """Read the deterministic PCM fixture without creating an intermediate WAV."""
    text = Path(path).read_text(encoding='utf-8')
    body = re.search(r'test_audio_pcm\s*\[[^]]+\]\s*=\s*\{(.*?)\};', text, re.S)
    if not body: raise ValueError('test_audio_pcm initializer not found')
    pcm = [int(x) for x in re.findall(r'-?\d+', body.group(1))]
    if len(pcm) != SAMPLES: raise ValueError(f'fixture has {len(pcm)}, expected {SAMPLES} samples')
    return pcm

def round_away(x): return math.floor(x + .5) if x >= 0 else math.ceil(x - .5)
def hz_to_mel(f): return 1127.0 * math.log(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (math.exp(m / 1127.0) - 1.0)

def pipeline(pcm):
    hann = [.5 * (1.0 - math.cos(2.0 * math.pi * n / NFFT)) for n in range(NFFT)]
    lo, hi = hz_to_mel(0), hz_to_mel(SR / 2)
    bins = [math.floor((NFFT + 1) * mel_to_hz(lo + (hi - lo) * i / (MELS + 1)) / SR) for i in range(MELS + 2)]
    def frame_db(frame):
        x = [pcm[frame * HOP + n] / 32768.0 * hann[n] for n in range(NFFT)]
        power = []
        for k in range(NFFT // 2 + 1):
            re = im = 0.0
            for n, value in enumerate(x):
                a = -2.0 * math.pi * k * n / NFFT
                re += value * math.cos(a); im += value * math.sin(a)
            power.append(re * re + im * im)
        values = []
        for m in range(MELS):
            left, center, right = bins[m:m+3]; energy = 0.0
            for k in range(left, min(right, NFFT // 2) + 1):
                weight = ((k-left)/(center-left) if k < center and center != left else
                          (right-k)/(right-center) if center <= k < right and right != center else 0.0)
                energy += power[k] * weight
            values.append(10.0 * math.log10(max(energy, 1e-10)))
        return values
    raw = [frame_db(f) for f in range(FRAMES)]
    global_max = max(max(row) for row in raw)
    mfcc = [0.0] * (MFCCS * FRAMES)
    q = [0] * len(mfcc)
    for f, row in enumerate(raw):
        row = [max(v, global_max - 80.0) - global_max for v in row]
        for c in range(MFCCS):
            scale = math.sqrt(1.0/MELS) if c == 0 else math.sqrt(2.0/MELS)
            value = scale * sum(row[m] * math.cos(math.pi*c*(2*m+1)/(2*MELS)) for m in range(MELS))
            i = c * FRAMES + f; mfcc[i] = value
            q[i] = max(-128, min(127, round_away(((value - MEAN) / STD) / SCALE) + ZP))
    return mfcc, q, bins, global_max

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio', nargs='?'); ap.add_argument('--fixture-header'); ap.add_argument('--out', default='artifacts/reference')
    a = ap.parse_args()
    if bool(a.audio) == bool(a.fixture_header): ap.error('provide exactly one of audio or --fixture-header')
    pcm = read_pcm(a.audio) if a.audio else read_c_header(a.fixture_header)
    mfcc, q, bins, peak = pipeline(pcm); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.mfcc_f32le').write_bytes(struct.pack('<%df' % len(mfcc), *mfcc))
    out.with_suffix('.input_i8').write_bytes(struct.pack('<%db' % len(q), *q))
    def stats(values):
        mean = sum(values) / len(values)
        return {'min': min(values), 'max': max(values), 'mean': mean, 'std': math.sqrt(sum((x-mean)**2 for x in values)/len(values))}
    out.with_suffix('.json').write_text(json.dumps({'pipeline': 'active-main.cpp candidate', 'shape': [1,40,49,1], 'layout': 'mfcc-major', 'sample_rate': SR, 'frame_length': NFFT, 'hop_length': HOP, 'frames': FRAMES, 'mels': MELS, 'mfccs': MFCCS, 'window': 'periodic_hann', 'fft': 'unnormalised_dft_480', 'mel': 'htk_bin_rounded_unnormalised', 'top_db': 80.0, 'pcm': stats(pcm), 'mfcc': stats(mfcc), 'int8': stats(q), 'model_input_quantization': {'scale': SCALE, 'zero_point': ZP}, 'mel_bins': bins, 'global_max_db': peak, 'mfcc_file': str(out.with_suffix('.mfcc_f32le')), 'input_file': str(out.with_suffix('.input_i8'))}, indent=2))
    print('wrote', out.with_suffix('.mfcc_f32le'), out.with_suffix('.input_i8'), out.with_suffix('.json'))
if __name__ == '__main__': main()
