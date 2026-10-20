#!/usr/bin/env python3
"""
compare_pipeline.py
===================
PC-side reference script that exactly replicates the ESP32 MFCC
preprocessing pipeline and compares each stage against the values
printed by run_pipeline_diagnostic() on the device.

HOW TO USE
----------
1. Flash firmware with DIAG_MODE=1 and monitor serial:
       idf.py -p COM13 flash monitor

2. After capture completes, copy every "PCM_RAW[...]" log line from
   the monitor into the pcm_raw_lines list below, OR save the full
   monitor output to a file and run with --logfile path/to/monitor.log

3. Paste the ESP32 MFCC values (STAGE 5) into esp32_mfcc_frame0 and
   the INT8 values (STAGE 6) into esp32_int8_frame0.

4. Run:
       python compare_pipeline.py

PARAMETERS (must match firmware exactly)
-----------------------------------------
SAMPLE_RATE = 16000
FRAME_LENGTH = 480
HOP_LENGTH = 320
N_MELS = 40
N_MFCC = 40
NUM_FRAMES = 49
MODEL_SCALE = 0.0480119027197361
MODEL_ZP = 49
"""

import re
import sys
import math
import argparse
import numpy as np

# ============================================================
# STEP 1 — Paste the PCM_RAW lines from the serial monitor here
#          OR provide a log file with --logfile.
#
# Each line should look like:
#   PCM_RAW[1200,1195,1188,...,1240]
# ============================================================

pcm_raw_lines = [
    # PASTE LINES HERE, e.g.:
    # "PCM_RAW[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]",
]

# ============================================================
# STEP 2 — Paste ESP32 STAGE 5 MFCC values here for comparison.
#          Set to None to skip comparison.
# Format: one float per line from MFCC[00][frame0] to MFCC[39][frame0]
# ============================================================
esp32_mfcc_frame0 = None   # e.g. np.array([-323.313, ...])

# ============================================================
# STEP 3 — Paste ESP32 STAGE 6 int8 values here.
# ============================================================
esp32_int8_frame0 = None   # e.g. np.array([-128, 127, ...])


# ============================================================
# PIPELINE CONSTANTS (must match firmware)
# ============================================================

SAMPLE_RATE  = 16000
FRAME_LENGTH = 480
HOP_LENGTH   = 320
N_MELS       = 40
N_MFCC       = 40
NUM_FRAMES   = 49
MODEL_SCALE  = 0.0480119027197361
MODEL_ZP     = 49
AMIN         = 1e-10
TOP_DB       = 80.0
DIAG_FRAME   = 0     # which frame to inspect in detail


# ============================================================
# UTILITIES
# ============================================================

def parse_pcm_from_lines(lines):
    """Parse PCM_RAW[...] log lines into a flat int16 numpy array."""
    samples = []
    for line in lines:
        m = re.search(r'PCM_RAW\[([^\]]+)\]', line)
        if m:
            vals = [int(v.strip()) for v in m.group(1).split(',') if v.strip()]
            samples.extend(vals)
    return np.array(samples, dtype=np.int16)


def parse_pcm_from_logfile(path):
    """Read a saved serial monitor log file and extract PCM_RAW data."""
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()
    return parse_pcm_from_lines(lines)


def hz_to_mel_slaney(hz):
    """Slaney Mel conversion (same as librosa with htk=False)."""
    f_sp = 200.0 / 3.0
    if hz < 1000.0:
        return hz / f_sp
    else:
        min_log_hz  = 1000.0
        min_log_mel = min_log_hz / f_sp
        logstep     = math.log(6.4) / 27.0
        return min_log_mel + math.log(hz / min_log_hz) / logstep


def mel_to_hz_slaney(mel):
    """Inverse Slaney Mel conversion."""
    f_sp = 200.0 / 3.0
    min_log_hz  = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep     = math.log(6.4) / 27.0
    if mel < min_log_mel:
        return mel * f_sp
    else:
        return min_log_hz * math.exp(logstep * (mel - min_log_mel))


def build_hann_window(N):
    """Periodic Hann: 0.5 - 0.5*cos(2*pi*n/N). Matches firmware."""
    return np.array([0.5 - 0.5 * math.cos(2.0 * math.pi * n / N)
                     for n in range(N)], dtype=np.float32)


def build_slaney_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """
    Continuous Slaney Mel triangular filterbank (librosa compatible).
    Returns weights matrix [n_mels, n_fft//2+1].
    """
    if fmax is None:
        fmax = sr / 2.0
    n_bins = n_fft // 2 + 1

    mel_min = hz_to_mel_slaney(fmin)
    mel_max = hz_to_mel_slaney(fmax)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = np.array([mel_to_hz_slaney(m) for m in mel_points])

    # FFT bin frequencies k * sr / n_fft
    fft_freqs = np.array([k * sr / n_fft for k in range(n_bins)])

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left   = hz_points[m]
        center = hz_points[m + 1]
        right  = hz_points[m + 2]
        enorm  = 2.0 / (right - left)

        for k in range(n_bins):
            f = fft_freqs[k]
            if f < left or f > right:
                continue
            if f <= center:
                weights[m, k] = (f - left) / (center - left) * enorm
            else:
                weights[m, k] = (right - f) / (right - center) * enorm

    return weights


def compute_power_spectrum(windowed):
    """
    Unnormalized power spectrum matching ESP32 firmware.
    np.fft.rfft uses the standard DFT convention (no normalization).
    power[k] = |FFT[k]|^2 = real^2 + imag^2
    """
    fft_out = np.fft.rfft(windowed, n=FRAME_LENGTH)
    return fft_out.real**2 + fft_out.imag**2


def run_pipeline(pcm_int16):
    """
    Full MFCC pipeline matching the ESP32 firmware exactly.
    Returns dict of intermediate results for frame DIAG_FRAME.
    """
    min_needed = (NUM_FRAMES - 1) * HOP_LENGTH + FRAME_LENGTH
    if len(pcm_int16) < min_needed:
        pcm_int16 = np.pad(pcm_int16, (0, min_needed - len(pcm_int16)))

    pcm_f = pcm_int16.astype(np.float32)
    hann  = build_hann_window(FRAME_LENGTH)
    mel_W = build_slaney_mel_filterbank(SAMPLE_RATE, FRAME_LENGTH, N_MELS)

    # ---- Diagnostic frame (frame DIAG_FRAME) ----
    frame_start = DIAG_FRAME * HOP_LENGTH
    frame_pcm   = pcm_f[frame_start : frame_start + FRAME_LENGTH]
    frame_norm  = frame_pcm / 32768.0    # firmware divides by 32768.0f
    frame_win   = frame_norm * hann

    frame_power = compute_power_spectrum(frame_win)

    mel_energy_frame = mel_W @ frame_power
    mel_safe_frame   = np.maximum(mel_energy_frame, AMIN)
    mel_logdb_frame  = 10.0 * np.log10(mel_safe_frame)

    # ---- Full spectrogram over all 49 frames ----
    mel_spec = np.zeros((N_MELS, NUM_FRAMES), dtype=np.float64)
    for fi in range(NUM_FRAMES):
        fs    = fi * HOP_LENGTH
        f_pcm = pcm_f[fs : fs + FRAME_LENGTH] / 32768.0
        f_win = f_pcm * hann
        f_pwr = compute_power_spectrum(f_win)
        mel_e = mel_W @ f_pwr
        mel_s = np.maximum(mel_e, AMIN)
        mel_spec[:, fi] = 10.0 * np.log10(mel_s)

    # ---- Global top_db clipping (single global max across entire 40x49) ----
    global_max  = mel_spec.max()
    floor_db    = global_max - TOP_DB
    mel_clipped = np.maximum(mel_spec, floor_db)

    # ---- Orthonormal DCT-II (matches scipy.fft.dct type=2 norm='ortho') ----
    try:
        from scipy.fft import dct as scipy_dct
        mfcc_all = np.zeros((N_MFCC, NUM_FRAMES), dtype=np.float64)
        for fi in range(NUM_FRAMES):
            mfcc_all[:, fi] = scipy_dct(mel_clipped[:, fi], type=2, norm='ortho')[:N_MFCC]
    except ImportError:
        # Fallback: manual orthonormal DCT-II
        print("scipy not found; using manual DCT-II")
        mfcc_all = np.zeros((N_MFCC, NUM_FRAMES), dtype=np.float64)
        for fi in range(NUM_FRAMES):
            x = mel_clipped[:, fi]
            for c in range(N_MFCC):
                scale = math.sqrt(1.0 / N_MELS) if c == 0 else math.sqrt(2.0 / N_MELS)
                total = 0.0
                for m in range(N_MELS):
                    total += x[m] * math.cos(math.pi * c * (m + 0.5) / N_MELS)
                mfcc_all[c, fi] = scale * total

    mfcc_frame = mfcc_all[:, DIAG_FRAME]

    # ---- Quantize to int8 ----
    q_float   = mfcc_all / MODEL_SCALE + MODEL_ZP
    q_int8    = np.clip(np.round(q_float), -128, 127).astype(np.int8)

    return {
        'frame_win':       frame_win,
        'frame_power':     frame_power,
        'mel_energy':      mel_energy_frame,
        'mel_logdb':       mel_logdb_frame,
        'mel_spec_raw':    mel_spec,
        'mel_spec':        mel_clipped,
        'global_max_db':   global_max,
        'floor_db':        floor_db,
        'mfcc_all':        mfcc_all,
        'mfcc_frame':      mfcc_frame,
        'q_float_frame':   q_float[:, DIAG_FRAME],
        'q_int8_all':      q_int8,
        'q_int8_frame':    q_int8[:, DIAG_FRAME],
    }


def print_sep(label=''):
    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
        print(f"{'='*60}")


def compare_arrays(label, py_vals, esp32_vals):
    if esp32_vals is None:
        return
    py_arr  = np.asarray(py_vals,  dtype=np.float64)
    esp_arr = np.asarray(esp32_vals, dtype=np.float64)
    if py_arr.shape != esp_arr.shape:
        print(f"[{label}] SHAPE MISMATCH: py={py_arr.shape} esp={esp_arr.shape}")
        return
    diff    = np.abs(py_arr - esp_arr)
    idx     = np.argmax(diff)
    print_sep(f"COMPARISON: {label}")
    print(f"  Max |diff|  = {diff.max():.8f}  at index {idx}")
    print(f"  Mean |diff| = {diff.mean():.8f}")
    print(f"  Python   [{idx}] = {py_arr[idx]:.8f}")
    print(f"  ESP32    [{idx}] = {esp_arr[idx]:.8f}")
    print(f"  All diffs:")
    for i, d in enumerate(diff):
        flag = '  <-- MAX' if i == idx else ''
        print(f"    [{i:2d}] py={py_arr[i]:12.6f}  esp32={esp_arr[i]:12.6f}  diff={d:.6f}{flag}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ESP32 MFCC pipeline stage comparator')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Path to saved serial monitor log file')
    args = parser.parse_args()

    # Collect PCM samples
    if args.logfile:
        print(f"Reading PCM from log file: {args.logfile}")
        pcm = parse_pcm_from_logfile(args.logfile)
    elif pcm_raw_lines:
        print("Parsing PCM from embedded pcm_raw_lines...")
        pcm = parse_pcm_from_lines(pcm_raw_lines)
    else:
        print("ERROR: No PCM data provided.")
        print("Either set pcm_raw_lines or use --logfile path/to/monitor.log")
        sys.exit(1)

    frame0 = pcm[:FRAME_LENGTH].astype(np.float64)
    print_sep("STAGE 0: RAW PCM FRAME 0 STATISTICS")
    print(f"  Samples loaded : {len(pcm)}")
    print(f"  Frame0 MIN     = {frame0.min():.0f}")
    print(f"  Frame0 MAX     = {frame0.max():.0f}")
    print(f"  Frame0 MEAN    = {frame0.mean():.4f}")
    print(f"  Frame0 RMS     = {np.sqrt(np.mean(frame0**2)):.4f}")
    dc_raw  = frame0.mean()
    print(f"  DC offset      = {dc_raw:.4f}  ({100*dc_raw/32768:.3f}% of FS)")
    print(f"  First 8 samples: {pcm[:8].tolist()}")

    res = run_pipeline(pcm)

    print_sep("STAGE 1: WINDOWED FRAME 0 (first 8 values)")
    for i in range(min(8, len(res['frame_win']))):
        print(f"  win[{i}] = {res['frame_win'][i]:.8f}")
    print(f"  WIN MIN={res['frame_win'].min():.8f}  MAX={res['frame_win'].max():.8f}  MEAN={res['frame_win'].mean():.8f}")

    print_sep("STAGE 2: FFT POWER SPECTRUM (spot bins)")
    spot_bins = [0, 1, 2, 10, 20, 50, 100, 200, 240]
    for k in spot_bins:
        print(f"  PWR[{k:3d}] = {res['frame_power'][k]:.6e}")
    print(f"  PWR MIN={res['frame_power'].min():.6e}  MAX={res['frame_power'].max():.6e}  MEAN={res['frame_power'].mean():.6e}")
    print(f"  DC energy (bin 0) = {res['frame_power'][0]:.6e}")

    print_sep("STAGE 3: MEL ENERGIES and LOG-DB (frame 0)")
    for m in range(N_MELS):
        print(f"  MEL[{m:02d}]  energy={res['mel_energy'][m]:.8e}  log_db={res['mel_logdb'][m]:.6f}")

    print_sep("STAGE 3b: GLOBAL MEL SPECTROGRAM")
    print(f"  global_max_db = {res['global_max_db']:.6f}")
    print(f"  floor_db      = {res['floor_db']:.6f}")
    print(f"  mel_spec MIN  = {res['mel_spec'].min():.6f}")
    print(f"  mel_spec MAX  = {res['mel_spec'].max():.6f}")
    print(f"  mel_spec MEAN = {res['mel_spec'].mean():.6f}")

    print_sep("STAGE 4: MFCC FRAME 0 (40 coefficients)")
    for c in range(N_MFCC):
        print(f"  MFCC[{c:02d}][frame0] = {res['mfcc_frame'][c]:.8f}")

    print_sep("STAGE 4: FULL MFCC TENSOR STATISTICS")
    print(f"  MIN  = {res['mfcc_all'].min():.6f}")
    print(f"  MAX  = {res['mfcc_all'].max():.6f}")
    print(f"  MEAN = {res['mfcc_all'].mean():.6f}")

    print_sep("STAGE 5: Q_FLOAT and INT8 FRAME 0")
    for c in range(N_MFCC):
        print(f"  Q[{c:02d}][frame0]  q_float={res['q_float_frame'][c]:8.4f}  int8={int(res['q_int8_frame'][c])}")

    print_sep("STAGE 5: FULL INT8 TENSOR STATISTICS")
    print(f"  MIN  = {int(res['q_int8_all'].min())}")
    print(f"  MAX  = {int(res['q_int8_all'].max())}")
    print(f"  MEAN = {res['q_int8_all'].astype(float).mean():.4f}")

    # Compare against ESP32 values if provided
    compare_arrays("MFCC frame0", res['mfcc_frame'], esp32_mfcc_frame0)
    compare_arrays("INT8 frame0", res['q_int8_frame'].astype(float), esp32_int8_frame0)

    print("\nDone. Review the stage values above and compare them to the ESP32 log output.")
    print("If the MFCC values match but INT8 values diverge, the issue is in quantize_input().")
    print("If MEL LOG-DB diverges, the issue is in the Mel filterbank or FFT normalization.")
    print("If FFT POWER diverges, the issue is in the windowing or FFT implementation.")


if __name__ == '__main__':
    main()
