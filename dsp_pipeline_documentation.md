# Pluto Wake Word System: Complete DSP Pipeline

This document outlines the exact, step-by-step Digital Signal Processing (DSP) pipeline that your voice goes through inside the ESP32 (on Core 0) before it is handed over to the Machine Learning model (on Core 1).

---

### 1. I2S Hardware Audio Capture
* **Hardware:** INMP441 I2S Microphone
* **Sample Rate:** `16000 Hz` (16 kHz)
* **Bit Depth:** Captured as `32-bit` (Left-justified Philips format), downmixed via software bit-shifting (`>> 16`) to standard `16-bit PCM`.
* **Channel:** The I2S hardware is forced into `I2S_SLOT_MODE_STEREO` to bypass an ESP32 hardware clock bug, but the software specifically parses the interleaved data to extract **only the Left channel** (mono).
* **Chunk Size:** Reads `320 samples` per chunk (exactly 20 milliseconds of audio).

### 2. The Sliding "Audio Window"
* **Size:** `16,000 samples` (Exactly 1 second of audio history).
* **Mechanism:** Every 20 milliseconds, the oldest 320 samples are deleted from the front of the array, and the 320 brand-new samples are appended to the back using a `memmove()` operation. This guarantees the ML model is always looking at the most recent 1 second of continuous audio.

### 3. Pre-Emphasis Filter & Windowing
To prepare the 30-millisecond audio chunk for the Fourier Transform, two things happen:
* **Pre-Emphasis Filter:** Human speech naturally drops off at higher frequencies. A pre-emphasis filter (`0.97` coefficient) boosts the high frequencies to flatten the spectrum so the ML model can see consonants better.
* **Hann Window:** The audio chunk is multiplied by a mathematical bell curve (`0.5 * (1 - cos(2*pi*n / 479))`). This tapers the edges of the 30ms chunk to pure silence. If you don't do this, the FFT algorithm will invent fake high-frequency noise (spectral leakage) at the hard boundaries of the chunk.

### 4. Fast Fourier Transform (FFT)
* **Size:** `480-point Real FFT`
* **Operation:** Converts the 480 audio samples from the "Time Domain" (a raw waveform) into the "Frequency Domain" (a power spectrum showing how loud each individual frequency is).

### 5. Mel Filterbank
* **Filters:** `40 Bins`
* **Operation:** Human ears don't hear frequencies linearly; we are very good at distinguishing low pitches but terrible at distinguishing high pitches. The ESP32 maps the raw FFT power spectrum onto the non-linear "Mel Scale" using 40 overlapping triangular filters.
* **Output:** It computes the Log10 energy for each of the 40 filters (measured in Decibels).

### 6. The Mel Ring Buffer
* **Storage:** The 40 Mel features are pushed into a 2D circular array: `float mel_ring_buffer[49][40]`
* **History:** It stores exactly `49 frames` (980 milliseconds). 
* **The Hand-off:** Every 100 milliseconds (5 frames), if the audio energy is high enough, Core 0 sends an RTOS Task Notification to wake up Core 1.

### 7. Core 1: Machine Learning Prep & Inference
* **Normalization:** Core 1 wakes up and normalizes the 40 Mel features using the exact Mean (`-5.21585`) and Standard Deviation (`46.2198`) that the model was originally trained on in Python.
* **Quantization:** The floating-point numbers are squished into `INT8` (8-bit integers from -128 to +127) using the dynamic scaling factor built into your specific TFLite model.
* **The Neural Network:** The final `[1, 49, 40, 1]` INT8 tensor is injected into the Tiny DS-CNN model. If the output probability for class 0 is > `92.0%`, the system triggers the "Hey Pluto" LED and WebSocket!
