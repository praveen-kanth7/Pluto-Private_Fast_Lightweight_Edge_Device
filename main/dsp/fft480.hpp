#pragma once

#define FFT_SIZE 480

// Initializes the fast 480-point FFT tables
void fft480_init();

// Computes the power spectrum of the input
// input: array of length FFT_SIZE
// power_spectrum: output array of length FFT_SIZE/2 + 1
void fft480_compute_power(const float *input, float *power_spectrum);
