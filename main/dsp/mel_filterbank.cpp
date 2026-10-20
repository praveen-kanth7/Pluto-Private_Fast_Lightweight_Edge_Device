#include "mel_filterbank.hpp"
#include <math.h>
#include <string.h>

static float mel_bank[MEL_FILTERS][FFT_SIZE / 2 + 1];
static bool initialized = false;

static float hz_to_mel(float hz) {
    const float f_sp = 200.0f / 3.0f;
    const float min_hz = 1000.0f;
    const float min_mel = min_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (hz < min_hz) {
        return hz / f_sp;
    }
    return min_mel + logf(hz / min_hz) / logstep;
}

static float mel_to_hz(float mel) {
    const float f_sp = 200.0f / 3.0f;
    const float min_hz = 1000.0f;
    const float min_mel = min_hz / f_sp;
    const float logstep = logf(6.4f) / 27.0f;
    if (mel < min_mel) {
        return mel * f_sp;
    }
    return min_hz * expf(logstep * (mel - min_mel));
}

void mel_filterbank_init() {
    if (initialized) return;

    memset(mel_bank, 0, sizeof(mel_bank));
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(SAMPLE_RATE_F / 2.0f);
    
    float mel_points[MEL_FILTERS + 2];
    float hz_points[MEL_FILTERS + 2];
    
    for (int i = 0; i < MEL_FILTERS + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(MEL_FILTERS + 1);
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    for (int m = 0; m < MEL_FILTERS; ++m) {
        float lower = hz_points[m];
        float center = hz_points[m + 1];
        float upper = hz_points[m + 2];
        float norm = 2.0f / (upper - lower);
        
        for (int k = 0; k <= FFT_SIZE / 2; ++k) {
            float frequency = (float)k * SAMPLE_RATE_F / (float)FFT_SIZE;
            float weight = 0.0f;
            
            if (frequency > lower && frequency < center) {
                weight = (frequency - lower) / (center - lower);
            } else if (frequency >= center && frequency < upper) {
                weight = (upper - frequency) / (upper - center);
            }
            
            if (weight > 0.0f) {
                weight *= norm;
            }
            mel_bank[m][k] = weight;
        }
    }
    initialized = true;
}

float mel_filterbank_get_weight(int m, int k) {
    if (m >= 0 && m < MEL_FILTERS && k >= 0 && k <= FFT_SIZE / 2) {
        return mel_bank[m][k];
    }
    return 0.0f;
}
