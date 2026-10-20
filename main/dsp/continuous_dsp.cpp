#include "continuous_dsp.hpp"
#include "../ui/oled_display.hpp"
#include "fft480.hpp"
#include "mel_filterbank.hpp"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

static const char *TAG = "ContinuousDSP";

volatile bool trigger_wav_save = false;

#define I2S_NUM       I2S_NUM_0
#define I2S_BCLK      GPIO_NUM_4
#define I2S_WS        GPIO_NUM_5
#define I2S_SD        GPIO_NUM_6

#define AUDIO_SAMPLE_RATE 16000
#define FRAME_SIZE 480
#define FRAME_HOP  320
#define PI 3.14159265358979323846f
#define EPSILON 1e-10f

// Threshold of 400 RMS: filters ambient noise, HVAC, fan hum, general room noise.
// The INMP441 typical noise floor is 60-200 RMS; speech is typically 500-5000+ RMS.
// Raise this value if your room is louder (e.g. 600-800 for noisy environments).
#define ENERGY_THRESHOLD 400.0f

// Software digital gain. Set to 1.0f to match training data. High values cause severe clipping and destroy neural network accuracy!
#define MIC_GAIN 1.0f

TaskHandle_t inference_task_handle = nullptr;

float mel_ring_buffer[MFCC_FRAMES][MEL_FILTERS];
volatile int mel_ring_head = 0;
volatile float global_dsp_utilization = 0.0f;

#define HISTORY_SAMPLES 16000
static int16_t audio_history[HISTORY_SAMPLES] = {0};
static int history_head = 0;

static i2s_chan_handle_t rx_handle = NULL;
i2s_chan_handle_t rx_handle_global = NULL; // Exposed for ws_streamer.cpp
static float window_func[FRAME_SIZE];

static void init_window() {
    for (int n = 0; n < FRAME_SIZE; ++n) {
        window_func[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float)n / (float)FRAME_SIZE);
    }
}

static void audio_dsp_task(void *arg) {
    ESP_LOGI(TAG, "AudioDSPTask started on Core %d", xPortGetCoreID());

    static int16_t audio_window[FRAME_SIZE] = {0};
    static float fft_input[FRAME_SIZE];
    static float power_spectrum[FFT_SIZE / 2 + 1];

    int frames_since_last_inference = 0;
    while (1) {
        if (trigger_wav_save) {
            trigger_wav_save = false;
            ESP_LOGW(TAG, "SAVING 1 SECOND 'HEY PLUTO' AUDIO TO SPIFFS...");
            FILE *f = fopen("/spiffs/mic_test.wav", "wb");
            if (f) {
                int sample_rate = AUDIO_SAMPLE_RATE;
                int data_size = HISTORY_SAMPLES * 2; // 1 second of 16-bit audio
                int byte_rate = sample_rate * 2;
                
                uint8_t header[44] = {
                    'R','I','F','F',
                    (uint8_t)((data_size + 36) & 0xFF), (uint8_t)(((data_size + 36) >> 8) & 0xFF), (uint8_t)(((data_size + 36) >> 16) & 0xFF), (uint8_t)(((data_size + 36) >> 24) & 0xFF),
                    'W','A','V','E',
                    'f','m','t',' ',
                    16, 0, 0, 0,
                    1, 0, 1, 0,
                    (uint8_t)(sample_rate & 0xFF), (uint8_t)((sample_rate >> 8) & 0xFF), (uint8_t)((sample_rate >> 16) & 0xFF), (uint8_t)((sample_rate >> 24) & 0xFF),
                    (uint8_t)(byte_rate & 0xFF), (uint8_t)((byte_rate >> 8) & 0xFF), (uint8_t)((byte_rate >> 16) & 0xFF), (uint8_t)((byte_rate >> 24) & 0xFF),
                    2, 0, 16, 0,
                    'd','a','t','a',
                    (uint8_t)(data_size & 0xFF), (uint8_t)((data_size >> 8) & 0xFF), (uint8_t)((data_size >> 16) & 0xFF), (uint8_t)((data_size >> 24) & 0xFF)
                };
                fwrite(header, 1, 44, f);
                
                int tail = history_head;
                int first_chunk_size = HISTORY_SAMPLES - tail;
                fwrite(&audio_history[tail], sizeof(int16_t), first_chunk_size, f);
                if (tail > 0) {
                    fwrite(&audio_history[0], sizeof(int16_t), tail, f);
                }
                
                fclose(f);
                ESP_LOGW(TAG, "FINISHED RECORDING! Saved to /spiffs/mic_test.wav");
            }
        }

        // Because I2S is in STEREO mode, we must read 2 channels (L+R) and throw away R.
        // We read 640 stereo samples to get 320 mono samples.
        int32_t stereo_buffer[FRAME_HOP * 2];
        int32_t raw_buffer[FRAME_HOP];
        size_t chunk_bytes_read = 0;
        
        esp_err_t ret = i2s_channel_read(
            rx_handle,
            stereo_buffer,
            (FRAME_HOP * 2) * sizeof(int32_t),
            &chunk_bytes_read,
            portMAX_DELAY
        );

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "I2S read failed");
            continue;
        }

        int stereo_samples_read = chunk_bytes_read / sizeof(int32_t);
        int samples_read = stereo_samples_read / 2;
        
        // Extract Left channel (Even indices)
        for(int i = 0; i < samples_read; i++) {
            raw_buffer[i] = stereo_buffer[i * 2];
        }

        // Start timing ALL CPU work (excludes I2S read which is a hardware blocking wait)
        int64_t dsp_start = esp_timer_get_time();

        // Shift audio window left by FRAME_HOP (320)
        memmove(audio_window, &audio_window[FRAME_HOP], (FRAME_SIZE - FRAME_HOP) * sizeof(int16_t));

        // Calculate frame energy (RMS)
        float frame_energy = 0.0f;
        for (int i = 0; i < samples_read; i++) {
            float v = (float)(raw_buffer[i] >> 16);
            frame_energy += v * v;
        }
        frame_energy = sqrtf(frame_energy / samples_read);

        static int dbg_energy_counter = 0;
        if (++dbg_energy_counter >= 50) {
            ESP_LOGI(TAG, "[DEBUG] Room frame_energy RMS = %.2f", frame_energy);
            dbg_energy_counter = 0;
        }

        // Append new samples with Noise Gate

        // Convert to 16-bit PCM (NO NOISE GATE to prevent chopping the audio into static)
        for (int i = 0; i < samples_read; i++) {
            float val = (float)(raw_buffer[i] >> 16);
            
            // Hard clipping to prevent 16-bit overflow
            if (val > 32767.0f) val = 32767.0f;
            if (val < -32768.0f) val = -32768.0f;
            audio_window[(FRAME_SIZE - FRAME_HOP) + i] = (int16_t)val;
            
            audio_history[history_head] = (int16_t)val;
            history_head = (history_head + 1) % HISTORY_SAMPLES;
        }

        float current_mel_db[MEL_FILTERS];

        if (frame_energy < ENERGY_THRESHOLD) {
            oled_set_state(OLED_STATE_SLEEP);
            // VAD DSP Skip: The room is quiet.
            // Fill the mel ring with silence so the model sees a clean slate when speech starts.
            for (int m = 0; m < MEL_FILTERS; ++m) {
                current_mel_db[m] = -100.0f;
            }
        } else {
            oled_set_state(OLED_STATE_LISTENING);
            // Apply Hann window and normalize to [-1.0, 1.0]
            for (int n = 0; n < FRAME_SIZE; ++n) {
                float sample = (float)audio_window[n] / 32768.0f;
                fft_input[n] = sample * window_func[n];
            }

            // Compute FFT
            fft480_compute_power(fft_input, power_spectrum);

            // Compute Mel energies
            for (int m = 0; m < MEL_FILTERS; ++m) {
                float energy = 0.0f;
                for (int k = 0; k <= FFT_SIZE / 2; ++k) {
                    energy += power_spectrum[k] * mel_filterbank_get_weight(m, k);
                }
                if (energy < EPSILON) energy = EPSILON;
                current_mel_db[m] = 10.0f * log10f(energy);
            }
        }
        // Push to Ring Buffer
        int head = mel_ring_head;
        memcpy((void*)mel_ring_buffer[head], current_mel_db, MEL_FILTERS * sizeof(float));
        head = (head + 1) % MFCC_FRAMES;
        mel_ring_head = head;

        // End timing after all CPU work is done
        int64_t dsp_end = esp_timer_get_time();
        static int64_t total_dsp_time = 0;
        total_dsp_time += (dsp_end - dsp_start);

        static int dsp_log_counter = 0;
        dsp_log_counter++;
        if (dsp_log_counter >= 50) { // Log every 1 second (50 frames * 20ms)
            global_dsp_utilization = (float)total_dsp_time * 100.0f / (50.0f * 20000.0f);
            oled_set_cpu_utilization(global_dsp_utilization, 0.0f); // Update OLED in sleep too
            dsp_log_counter = 0;
            total_dsp_time = 0;
        }

        // CRITICAL: Only notify the Inference Task when there is actual speech energy.
        // In silence, the inference task stays blocked on ulTaskNotifyTake() consuming ~0% CPU.
        frames_since_last_inference++;
        if (frames_since_last_inference >= 5) {
            frames_since_last_inference = 0;
            // Only wake up Core 1 if we detected real audio energy above the noise gate
            if (frame_energy >= ENERGY_THRESHOLD && inference_task_handle != nullptr) {
                xTaskNotifyGive(inference_task_handle);
            }
        }
    }
}

void continuous_dsp_init() {
    ESP_LOGI(TAG, "Initializing INMP441...");

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    // CRITICAL FOR KWS: Massive DMA buffer prevents audio stuttering when the CPU is busy!
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 1024;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));
    rx_handle_global = rx_handle; // Share with ws_streamer.cpp

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE), // KWS usually requires 16000Hz
        // CRITICAL: Must be 32-bit and STEREO to generate the correct clock and bypass mono bugs
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK,
            .ws = I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_SD,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false
            }
        }
    };

    // Force the DMA buffer to only capture the Left channel so our DSP math doesn't break
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    fft480_init();
    mel_filterbank_init();
    init_window();

    xTaskCreatePinnedToCore(
        audio_dsp_task,
        "AudioDSP",
        8192,
        NULL,
        configMAX_PRIORITIES - 1, // High priority for audio
        NULL,
        0 // Core 0
    );
}
