#include "mfcc_processor.hpp"
#include "fft480.hpp"
#include "mel_filterbank.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include <math.h>

static const char *TAG = "DSP";

#define FRAME_SIZE 480
#define FRAME_HOP  320
#define PI 3.14159265358979323846f
#define EPSILON 1e-10f
#define TOP_DB 80.0f
#define TRAIN_MEAN (-5.2158537f)
#define TRAIN_STD  (46.219826f)
#define INPUT_SCALE 0.0480119027197361f
#define INPUT_ZERO 49

static QueueHandle_t in_pcm;
static QueueHandle_t free_pcm;
static QueueHandle_t out_feature;
static QueueHandle_t free_feature;

static float window_func[FRAME_SIZE];
static float dct_table[MFCC_COEFFS][MEL_FILTERS];

static void init_window() {
    for (int n = 0; n < FRAME_SIZE; ++n) {
        window_func[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float)n / (float)FRAME_SIZE);
    }
}

static void init_dct() {
    const float scale0 = sqrtf(1.0f / (float)MEL_FILTERS);
    const float scale = sqrtf(2.0f / (float)MEL_FILTERS);
    for (int c = 0; c < MFCC_COEFFS; ++c) {
        for (int m = 0; m < MEL_FILTERS; ++m) {
            float scale_c = (c == 0) ? scale0 : scale;
            dct_table[c][m] = scale_c * cosf(PI * (float)c * ((float)m + 0.5f) / (float)MEL_FILTERS);
        }
    }
}

static void process_frame(const AudioFrame *pcm, FeatureTensor *feature) {
    // Two-pass approach to avoid large arrays on the stack or in static memory.
    static float fft_input[FRAME_SIZE];
    static float power_spectrum[FFT_SIZE / 2 + 1];

    // PASS 1: Find the global maximum of the log-mel energies
    float global_max = -INFINITY;

    for (int frame = 0; frame < MFCC_FRAMES; ++frame) {
        int start = frame * FRAME_HOP;
        for (int n = 0; n < FRAME_SIZE; ++n) {
            float sample = (float)pcm->samples[start + n] / 32768.0f;
            fft_input[n] = sample * window_func[n];
        }
        
        fft480_compute_power(fft_input, power_spectrum);
        
        for (int m = 0; m < MEL_FILTERS; ++m) {
            float energy = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; ++k) {
                energy += power_spectrum[k] * mel_filterbank_get_weight(m, k);
            }
            if (energy < EPSILON) energy = EPSILON;
            
            float db = 10.0f * log10f(energy);
            if (db > global_max) {
                global_max = db;
            }
        }
    }
    
    // PASS 2: Recompute and calculate the final MFCC features directly to output
    for (int frame = 0; frame < MFCC_FRAMES; ++frame) {
        int start = frame * FRAME_HOP;
        for (int n = 0; n < FRAME_SIZE; ++n) {
            float sample = (float)pcm->samples[start + n] / 32768.0f;
            fft_input[n] = sample * window_func[n];
        }
        
        fft480_compute_power(fft_input, power_spectrum);
        
        float current_mel_db[MEL_FILTERS];
        for (int m = 0; m < MEL_FILTERS; ++m) {
            float energy = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; ++k) {
                energy += power_spectrum[k] * mel_filterbank_get_weight(m, k);
            }
            if (energy < EPSILON) energy = EPSILON;
            
            float db = 10.0f * log10f(energy);
            float floor_db = global_max - TOP_DB;
            if (db < floor_db) {
                db = floor_db;
            }
            current_mel_db[m] = db;
        }
        
        for (int c = 0; c < MFCC_COEFFS; ++c) {
            float sum = 0.0f;
            for (int m = 0; m < MEL_FILTERS; ++m) {
                sum += current_mel_db[m] * dct_table[c][m];
            }
            
            float normalized = (sum - TRAIN_MEAN) / TRAIN_STD;
            int q = (int)roundf(normalized / INPUT_SCALE) + INPUT_ZERO;
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            
            feature->data[c * MFCC_FRAMES + frame] = (int8_t)q;
        }
    }
}

static void dsp_task(void *arg) {
    ESP_LOGI(TAG, "DSPTask started on Core %d", xPortGetCoreID());

    while (1) {
        AudioFrame *pcm = nullptr;
        if (xQueueReceive(in_pcm, &pcm, portMAX_DELAY) == pdTRUE) {
            FeatureTensor *feature = nullptr;
            if (xQueueReceive(free_feature, &feature, 0) == pdTRUE) {
                
                int64_t start = esp_timer_get_time();
                process_frame(pcm, feature);
                int64_t end = esp_timer_get_time();
                
                ESP_LOGI(TAG, "MFCC processed in %lld us | HighWaterMark: %d words", 
                         end - start, uxTaskGetStackHighWaterMark(NULL));
                
                xQueueSend(out_feature, &feature, 0);
            } else {
                ESP_LOGE(TAG, "Feature queue full! Dropping DSP result.");
            }
            xQueueSend(free_pcm, &pcm, 0);
        }
    }
}

void dsp_processor_init(QueueHandle_t pcm_queue, QueueHandle_t pcm_free_queue,
                        QueueHandle_t feature_queue, QueueHandle_t feature_free_queue) {
    in_pcm = pcm_queue;
    free_pcm = pcm_free_queue;
    out_feature = feature_queue;
    free_feature = feature_free_queue;

    fft480_init();
    mel_filterbank_init();
    init_window();
    init_dct();

    xTaskCreatePinnedToCore(
        dsp_task,
        "DSP",
        6144, // 6KB stack should be enough since large arrays are static
        NULL,
        configMAX_PRIORITIES - 2, // Medium-High priority
        NULL,
        1 // Core 1
    );
}
