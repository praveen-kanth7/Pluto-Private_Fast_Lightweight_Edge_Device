#pragma once

#define MEL_FILTERS 40
#define FFT_SIZE 480
#define SAMPLE_RATE_F 16000.0f

void mel_filterbank_init();
float mel_filterbank_get_weight(int m, int k);
