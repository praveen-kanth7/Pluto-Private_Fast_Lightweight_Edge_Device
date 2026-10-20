#include "fft480.hpp"
#include <math.h>
#include <string.h>

#define PI 3.14159265358979323846f

static float w32r[16];
static float w32i[16];

static float w15r[15][15];
static float w15i[15][15];

static float w480r[32][15];
static float w480i[32][15];

static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];
static float stage_real[FFT_SIZE];
static float stage_imag[FFT_SIZE];

static bool initialized = false;

static void fft32(float *real, float *imag) {
    int j = 0;
    for (int i = 1; i < 32; ++i) {
        int bit = 16;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            float t;
            t = real[i]; real[i] = real[j]; real[j] = t;
            t = imag[i]; imag[i] = imag[j]; imag[j] = t;
        }
    }
    for (int len = 2; len <= 32; len <<= 1) {
        int half = len >> 1;
        for (int base = 0; base < 32; base += len) {
            for (int jj = 0; jj < half; ++jj) {
                int table_index = jj * 16 / half;
                float wr = w32r[table_index];
                float wi = w32i[table_index];
                float ar = real[base + jj];
                float ai = imag[base + jj];
                float br = real[base + jj + half];
                float bi = imag[base + jj + half];
                float tr = br * wr - bi * wi;
                float ti = br * wi + bi * wr;
                real[base + jj] = ar + tr;
                imag[base + jj] = ai + ti;
                real[base + jj + half] = ar - tr;
                imag[base + jj + half] = ai - ti;
            }
        }
    }
}

void fft480_init() {
    if (initialized) return;

    for (int k = 0; k < 16; ++k) {
        float a = -2.0f * PI * (float)k / 32.0f;
        w32r[k] = cosf(a);
        w32i[k] = sinf(a);
    }
    for (int k = 0; k < 15; ++k) {
        for (int n = 0; n < 15; ++n) {
            float a = -2.0f * PI * (float)k * (float)n / 15.0f;
            w15r[k][n] = cosf(a);
            w15i[k][n] = sinf(a);
        }
    }
    for (int k2 = 0; k2 < 32; ++k2) {
        for (int n1 = 0; n1 < 15; ++n1) {
            float a = -2.0f * PI * (float)k2 * (float)n1 / 480.0f;
            w480r[k2][n1] = cosf(a);
            w480i[k2][n1] = sinf(a);
        }
    }
    initialized = true;
}

void fft480_compute_power(const float *input, float *power_spectrum) {
    for (int n1 = 0; n1 < 15; ++n1) {
        for (int n2 = 0; n2 < 32; ++n2) {
            int index = n1 + 15 * n2;
            fft_real[n2] = input[index];
            fft_imag[n2] = 0.0f;
        }
        fft32(fft_real, fft_imag);
        for (int k2 = 0; k2 < 32; ++k2) {
            stage_real[n1 + 15 * k2] = fft_real[k2];
            stage_imag[n1 + 15 * k2] = fft_imag[k2];
        }
    }
    for (int k2 = 0; k2 < 32; ++k2) {
        for (int k1 = 0; k1 < 15; ++k1) {
            float sum_real = 0.0f;
            float sum_imag = 0.0f;
            for (int n1 = 0; n1 < 15; ++n1) {
                float ar = stage_real[n1 + 15 * k2];
                float ai = stage_imag[n1 + 15 * k2];
                float wr1 = w15r[k1][n1];
                float wi1 = w15i[k1][n1];
                float wr2 = w480r[k2][n1];
                float wi2 = w480i[k2][n1];
                float br = ar * wr1 - ai * wi1;
                float bi = ar * wi1 + ai * wr1;
                sum_real += br * wr2 - bi * wi2;
                sum_imag += br * wi2 + bi * wr2;
            }
            int output_index = k2 + 32 * k1;
            fft_real[output_index] = sum_real;
            fft_imag[output_index] = sum_imag;
        }
    }
    for (int k = 0; k <= FFT_SIZE / 2; ++k) {
        float re = fft_real[k];
        float im = fft_imag[k];
        power_spectrum[k] = re * re + im * im;
    }
}
