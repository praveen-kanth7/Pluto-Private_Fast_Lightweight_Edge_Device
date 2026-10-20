# KWS preprocessing audit — 2026-09-04

## Scope and evidence

`main/CMakeLists.txt` builds `main.cpp`; therefore this is the active extractor.
`main/mfcc.cpp` is an unused experimental implementation and was not previously
compiled. It must not be treated as evidence of the model training pipeline.

The repository contains no dataset, training notebook/script, or TFLite model
conversion script. `convert_model.py` only embeds an existing flatbuffer. Thus
the original Librosa calls and exact training parameters are **UNKNOWN — needs
to be measured/inspected** in the training project.

## Active deployed pipeline (`main.cpp`)

| Item | Verified value / behavior |
|---|---|
| Capture | INMP441 I2S Philips, 32-bit mono left slot, 16,000 Hz; `raw >> 16` produces PCM16 |
| Channels | Mono; no stereo-to-mono mixing occurs. |
| Analysis input | First 15,840 samples of a 16,000-sample capture; the last 160 samples are unused. |
| PCM scaling | `float(pcm16) / 32768.0f`; no peak/RMS/DC normalization. |
| Pre-emphasis | None. |
| Frames | 49, start `frame * 320`, equivalent to `center=false`. |
| Frame length / hop | 480 / 320 samples = 30 / 20 ms. |
| Window | Periodic Hann: `0.5*(1-cos(2*pi*n/480))`. |
| FFT | 480-point unnormalised DFT in active firmware; power `re²+im²`, 241 bins, 33.333 Hz/bin. |
| Mel filters | 40 triangles, 0–8,000 Hz; HTK `1127*ln(1+f/700)`, edges `floor(481*Hz/16000)`, no Slaney/area normalization. |
| Log | `10*log10(max(energy,1e-10))`, utterance-global maximum reference, 80 dB floor. |
| DCT | 40 DCT-II coefficients, orthonormal (`sqrt(1/40)` for c0, `sqrt(2/40)` otherwise). |
| Float feature | MFCC `[40,49]`, flattened `mfcc*49 + frame`. |
| Training normalization | `(mfcc - -5.2158537) / 46.219826`. |
| Input quantization | `roundf(normalized / 0.0480119027197361) + 49`, clamped INT8. |
| Model input | Verified from `ds_cnn_int8.tflite`: `[1,40,49,1]`, INT8, scale `0.0480119027197361`, zero point `49`, 1,960 bytes. |
| Model output | Verified from flatbuffer: `[1,3]`, INT8, scale `0.00390625`, zero point `-128`. |

The model expects **A. MFCC features**. The DCT output, not raw PCM or Log-Mel,
is written to tensor 0.

## Librosa status

There is no executable `import librosa` in the repository. Comments in the
active code claim Librosa compatibility, but its Mel-bank definition is not the
usual `librosa.filters.mel` default.

The unused `main/mfcc.cpp` instead claims periodic Hann, 480-point STFT, hop
320, 40 Slaney Mel filters (0–8 kHz, `htk=False`, `norm='slaney'`),
`power_to_db(ref=np.max, top_db=80)`, and orthonormal DCT-II. It is incompatible
with `main.cpp` in Mel scale, normalization, and bin interpolation, and omits
the mean/std input step. Do not use it as a training reference.

Every actual Librosa function/parameter affecting the dataset is **UNKNOWN —
needs to be measured/inspected**. Inspect the training source for `librosa.load`,
resampling, channel handling, STFT/Mel/MFCC calls, `power_to_db`, and numpy
normalization, and retain a WAV plus its saved training tensor.

## Reference, ESP32 candidate, and validation

`tools/reference_preprocess.py` is a dependency-free PC reference for the
**active firmware**, not an assertion about unavailable training code. It accepts
only exactly 16 kHz mono PCM16 to prevent implicit resampling/downmixing. It
writes 1,960 little-endian float32 MFCC values, 1,960 signed INT8 input bytes,
and metadata including shape/layout/bin edges.

`main/kws_preprocess.{h,cpp}` is an ESP-IDF-compatible candidate. It reproduces
active `main.cpp` math but uses two passes and writes directly to TFLM input
layout instead of retaining full MFCC/Mel matrices. It is compiled but not wired
into the inference loop: changing active detection before same-audio validation
would mask a training/deployment mismatch.

Inspect the checked-in model with `python tools/inspect_tflite.py`; it verifies
one subgraph, 30 tensors, 12 operators, and the model tensor contracts above.

Run after obtaining a raw capture and equivalent ESP32 artifact dumps:

```powershell
python tools/reference_preprocess.py esp32_capture.raw --out artifacts/reference
python tools/validate_preprocess.py --pc-mfcc artifacts/reference.mfcc_f32le --esp-mfcc artifacts/esp32.mfcc_f32le --pc-input artifacts/reference.input_i8 --esp-input artifacts/esp32.input_i8 --pc-output <p0> <p1> <p2> --esp-output <p0> <p1> <p2>
```

The validator reports max absolute difference, mean absolute difference,
relative error (mean absolute difference divided by mean absolute PC value),
and equal argmax prediction. No same-audio ESP32 dump or PC TFLite runtime is
present, so actual metrics and prediction agreement are **UNKNOWN — needs to
be measured/inspected**.

## RAM audit

Values are bytes from declared arrays, not whole-firmware heap/stack proof.

| Area | Current active `main.cpp` | Candidate, additional to input arena |
|---|---:|---:|
| PCM capture buffer | 32,000 | 32,000 (15,840 / 31,680 possible after capture change) |
| Tensor arena | 122,880 configured; true minimum **UNKNOWN — needs measurement** | same |
| FFT / power work | 4,804 (`fft_real`, `fft_imag`, power) | 964 stack power buffer |
| Mel bank/helpers | 39,040 (dense bank + bounds + energies) | 168 bin edges + 160 stack Mel values |
| MFCC / log-Mel retention | 7,840 MFCC static + 7,840 `mel_db` task stack | no full matrix |
| Hann | 1,920 | 1,920 |
| Preprocessor total | — | about 3,212 persistent/work bytes, excluding PCM and TFLM input |
| Active declared subtotal | **208,484** static bytes; **216,324** including the 7,840-byte `mel_db` task-stack local | — |

The current build does not prove a <256 KiB complete application: its declared
arrays/work already total 216,324 B before IDF, FreeRTOS stacks, I2S/DMA, TFLM
persistent allocation, logging, and app buffers. Measure the arena with a
`RecordingMicroInterpreter` or on-target binary search after `AllocateTensors()`;
the actual requirement is **UNKNOWN — needs to be measured/inspected**.

Candidate KWS-only accounting is 32,000 PCM + 122,880 arena + 3,212
preprocessing = about **158 KiB**, before TFLM/IDF/I2S/task overhead. It is
plausibly below 256 KiB but cannot be guaranteed without on-target measurement.
The 15,048-byte model is flash data through `EMBED_FILES`, not DRAM.

## CPU, latency, and feature removal

The active DFT executes 49×241×480 = 5,668,320 bin/sample iterations, each
with `sinf` and `cosf`. The candidate does two passes. Neither is realistic for
continuous <10% CPU, particularly with the current debug (`-Og`) build. The
design captures one second then waits 500 ms; it is not a sliding low-latency
detector.

MFCC/Log-Mel extraction cannot be removed for this model: convolution expects a
quantized MFCC image. Feeding PCM changes tensor shape and learned semantics;
a raw-waveform model needs retraining and may alter size, MACs, latency, and
accuracy. The practical reduction is the candidate's direct-to-input approach,
then a validated equivalent 480-point mixed-radix/fixed-point FFT and
sparse/quantized flash tables. This does not increase model size.
