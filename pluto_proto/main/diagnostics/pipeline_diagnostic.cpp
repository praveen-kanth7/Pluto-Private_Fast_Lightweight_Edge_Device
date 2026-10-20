#include "pipeline_diagnostic.hpp"
#include "../dsp/fft480.hpp"
#include "../dsp/mel_filterbank.hpp"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <math.h>
#include <stdio.h>
#include <algorithm>

static const char *TAG = "Diag";

#define AUDIO_SAMPLES 16000
#define FRAME_SIZE 480
#define FRAME_HOP 320
#define PI 3.14159265358979323846f
#define EPSILON 1e-10f
#define TOP_DB 80.0f
#define TRAIN_MEAN (-5.2158537f)
#define TRAIN_STD  (46.219826f)
#define INPUT_SCALE 0.0480119027197361f
#define INPUT_ZERO 49

static int16_t diag_audio_buffer[AUDIO_SAMPLES];

void pipeline_diagnostic_run() {
    ESP_LOGI(TAG, "Running DIAG_MODE=1");

    // Initialize I2S
    i2s_chan_handle_t rx_handle;
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED, .bclk = GPIO_NUM_4, .ws = GPIO_NUM_5, .dout = I2S_GPIO_UNUSED, .din = GPIO_NUM_6,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false }
        }
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "Capturing 1 second...");
    
    int samples_received = 0;
    while (samples_received < AUDIO_SAMPLES) {
        int32_t raw_buffer[250];
        size_t chunk_bytes_read = 0;
        int to_read = (AUDIO_SAMPLES - samples_received > 250) ? 250 : (AUDIO_SAMPLES - samples_received);
        i2s_channel_read(rx_handle, raw_buffer, to_read * sizeof(int32_t), &chunk_bytes_read, portMAX_DELAY);
        int samples_read = chunk_bytes_read / sizeof(int32_t);
        for (int i = 0; i < samples_read; i++) {
            diag_audio_buffer[samples_received + i] = (int16_t)(raw_buffer[i] >> 16);
        }
        samples_received += samples_read;
    }
    ESP_LOGI(TAG, "Capture complete.");

    printf("\nSTAGE 0: Raw PCM statistics and all 480 PCM samples\n");
    printf("PCM_RAW[");
    for (int i = 0; i < FRAME_SIZE; i++) {
        printf("%d%s", diag_audio_buffer[i], (i == FRAME_SIZE - 1) ? "" : ",");
    }
    printf("]\n");

    fft480_init();
    mel_filterbank_init();

    float window_func[FRAME_SIZE];
    for (int n = 0; n < FRAME_SIZE; ++n) {
        window_func[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float)n / (float)FRAME_SIZE);
    }

    float fft_input[FRAME_SIZE];
    float power_spectrum[480 / 2 + 1];
    float mel_energies[40];

    printf("\nSTAGE 1: Windowed frame statistics\n");
    for (int n = 0; n < FRAME_SIZE; ++n) {
        fft_input[n] = ((float)diag_audio_buffer[n] / 32768.0f) * window_func[n];
    }
    printf("WIN_RAW[");
    for (int n = 0; n < std::min(10, FRAME_SIZE); ++n) printf("%f,", fft_input[n]);
    printf("...]\n");

    printf("\nSTAGE 2: FFT power statistics\n");
    fft480_compute_power(fft_input, power_spectrum);
    printf("PWR_RAW[");
    for (int n = 0; n < std::min(10, 241); ++n) printf("%e,", power_spectrum[n]);
    printf("...]\n");

    printf("\nSTAGE 3: 40 mel energies and log-dB\n");
    for (int m = 0; m < 40; ++m) {
        float energy = 0.0f;
        for (int k = 0; k <= 240; ++k) {
            energy += power_spectrum[k] * mel_filterbank_get_weight(m, k);
        }
        if (energy < EPSILON) energy = EPSILON;
        mel_energies[m] = energy;
        printf("MEL[%d]=%e, dB=%f\n", m, energy, 10.0f * log10f(energy));
    }

    printf("\nSTAGE 4: MFCC pipeline\n");
    printf("\nSTAGE 5: 40 MFCC coefficients for frame 0\n");
    printf("\nSTAGE 6: 40 INT8 values\n");
    printf("\nSTAGE 7: Global mel statistics\n");

    ESP_LOGI(TAG, "Halting.");
    while (1) vTaskDelay(portMAX_DELAY);
}
