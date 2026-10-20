#include "inference_engine.hpp"
#include "wake_word_model.hpp"
#include "../ui/oled_display.hpp"
#include "../ui/rgb_led.hpp"
#include "../dsp/continuous_dsp.hpp"
#include "../ws_streamer.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <math.h>
#include <string.h>

static const char *TAG = "Inference";

extern volatile float global_dsp_utilization;
extern volatile bool trigger_wav_save;

// Tiny DS-CNN INT8 tensor arena — ESP32-S3 runtime sizing
// Desktop test_arena (host/reference kernels) measured: 43,088 bytes
// ESP32-S3 ESP-NN optimised kernels need ~20 KB extra scratch buffers:
//   AllocateTensors() requested 62,896 bytes, failed at 42,356 available
//   Minimum on-device: ~65,596 bytes (2,700 header + 62,896 allocation)
// 68 KB = 69,632 bytes → ~4 KB headroom over measured on-device minimum
// RAM saved vs old 128 KB arena: 60 KB
#define TENSOR_ARENA_SIZE (68 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

#define TOP_DB 80.0f
#define MFCC_COEFFS 40
#define TRAIN_MEAN (-5.2158537f)
#define TRAIN_STD  (46.219826f)
// INPUT_SCALE and INPUT_ZERO are read dynamically from the tensor after
// AllocateTensors() — do NOT hardcode these; they differ per model.
static float g_input_scale = 1.0f;
static int   g_input_zero  = 0;
#define PI 3.14159265358979323846f

static float dct_table[MFCC_COEFFS][MEL_FILTERS];

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

static void inference_task(void *arg) {
    ESP_LOGI(TAG, "InferenceTask started on Core %d", xPortGetCoreID());
    // Timestamp-based timers (microseconds) - advance in real time regardless of audio.
    // Frame-count timers freeze in silence since DSP only notifies when RMS >= 400.
    const int64_t STARTUP_DELAY_US = 3000000LL; // 3s after boot before detection enabled
    const int64_t LED_DURATION_US  = 3000000LL; // LED stays on for 3 seconds
    const int64_t COOLDOWN_US      = 3000000LL; // Re-trigger blocked for 3 seconds
    static int64_t led_off_at     = 0;          // Wall-clock time to turn LED off
    static int64_t cooldown_until = 0;          // Wall-clock time cooldown expires

    while (1) {
        // Wait for notification from DSP task, or wake up every 100ms to check LED timer.
        // Without a timeout, the task sleeps indefinitely in silence (DSP only notifies
        // when audio RMS >= 400), which prevents the LED from turning off after detection.
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));

        int64_t now = esp_timer_get_time();

        // Startup guard: real-time, always works even in silence
        if (now < STARTUP_DELAY_US) {
            continue;
        }

        // LED timer: turn off when duration has elapsed
        if (led_off_at > 0 && now >= led_off_at) {
            gpio_set_level(GPIO_NUM_2, 0);
            led_off_at = 0;
        }

        // 1. Read ring buffer and find global_max
        float local_ring[MFCC_FRAMES][MEL_FILTERS];
        int head = mel_ring_head;
        
        // Unroll ring buffer into chronological order
        float global_max = -INFINITY;
        for (int i = 0; i < MFCC_FRAMES; ++i) {
            int idx = (head + i) % MFCC_FRAMES;
            for (int m = 0; m < MEL_FILTERS; ++m) {
                float val = mel_ring_buffer[idx][m];
                local_ring[i][m] = val;
                if (val > global_max) {
                    global_max = val;
                }
            }
        }

        // VAD (Voice Activity Detection) Skip
        // If the entire 1-second window is pure silence, the DSP noise gate will have
        // squelched it. In that case, global_max is roughly -100 dB.
        
        static int dbg_vad_counter = 0;
        if (++dbg_vad_counter >= 10) {
            ESP_LOGI(TAG, "[DEBUG] global_max = %.2f dB", global_max);
            dbg_vad_counter = 0;
        }

        if (global_max < -90.0f) {
            static int vad_log = 0;
            if (++vad_log >= 10) {
                ESP_LOGI(TAG, "[PROFILER] CPU Utilization -> Core 0 (DSP): %.1f%% | Core 1 (ML): 0.0%% (VAD SLEEP) | Free RAM: %u bytes", global_dsp_utilization, esp_get_free_heap_size());
                oled_set_cpu_utilization(global_dsp_utilization, 0.0f);
                vad_log = 0;
            }
            continue; // Skip the heavy neural network entirely!
        }

        // Skip inference during startup to prevent boot-noise false detections
        // NOTE: This guard is now handled at the top of the loop (before VAD),
        // so by the time we reach here, startup_guard is already 0.

        // 2. Normalize and Quantize (Pass 2) directly into input_tensor
        float floor_db = global_max - TOP_DB;
        for (int frame = 0; frame < MFCC_FRAMES; ++frame) {
            for (int c = 0; c < MFCC_COEFFS; ++c) {
                float sum = 0.0f;
                for (int m = 0; m < MEL_FILTERS; ++m) {
                    float db = local_ring[frame][m];
                    if (db < floor_db) db = floor_db;
    
                    sum += db * dct_table[c][m];
                }
                
                float normalized = (sum - TRAIN_MEAN) / TRAIN_STD;
                // Use scale/zero read from this model's input tensor — NOT hardcoded
                int q = (int)roundf(normalized / g_input_scale) + g_input_zero;
                if (q < -128) q = -128;
                if (q > 127)  q = 127;
                
                // CRITICAL FIX: The model expects shape [1, 49, 40, 1] (Frames x MFCCs)
                // The index must be frame * 40 + c. The previous code was transposing the image!
                input_tensor->data.int8[frame * MFCC_COEFFS + c] = (int8_t)q;
            }
        }

        // 3. Run Inference
        int64_t start = esp_timer_get_time();
        TfLiteStatus status = interpreter->Invoke();
        int64_t end = esp_timer_get_time();

        if (status != kTfLiteOk) {
            ESP_LOGE(TAG, "TFLite Invoke failed");
        } else {
            static int log_counter = 0;
            log_counter++;
            if (log_counter >= 10) { // Log every 1 second (10 frames * 100ms)
                float ml_percent = (float)(end - start) * 100.0f / 100000.0f;
                ESP_LOGI(TAG, "[PROFILER] CPU Utilization -> Core 0 (DSP): %.1f%% | Core 1 (ML): %.1f%% | Free RAM: %u bytes", 
                         global_dsp_utilization, ml_percent, esp_get_free_heap_size());
                oled_set_cpu_utilization(global_dsp_utilization, ml_percent);
                log_counter = 0;
            }
            
            // Log EVERY frame during debugging so we can see raw model output
            int8_t raw0 = output_tensor->data.int8[0];
            int8_t raw1 = output_tensor->data.int8[1];

            float prob0 = ((int)raw0 - output_tensor->params.zero_point) * output_tensor->params.scale;
            float prob1 = ((int)raw1 - output_tensor->params.zero_point) * output_tensor->params.scale;

            int prediction_class = (prob1 > prob0) ? 1 : 0;

            // Handle all edge cases / test scenarios explicitly for debugging
            bool in_cooldown = (now < cooldown_until);
            
            if (prediction_class == 0) { // The model thinks it heard "Hey Pluto"
                if (prob0 > 0.92f) {
                    // Test Case 1: High confidence, but we are in cooldown
                    if (in_cooldown) {
                        ESP_LOGW(TAG, "Detection BLOCKED by cooldown (%.1f%%)", prob0 * 100.0f);
                    } 
                    // Test Case 2: High confidence, NOT in cooldown -> VALID TRIGGER
                    else {
                        ESP_LOGI(TAG, ">>> HEY PLUTO DETECTED <<< (Confidence: %.1f%%)", prob0 * 100.0f);
                        trigger_wav_save = true; // Tell the DSP thread to save the 1-second audio window!
                        oled_set_state(OLED_STATE_DETECTED);
                        ws_trigger_stream(now / 1000LL); // µs to ms
                        cooldown_until = now + COOLDOWN_US;
                        led_off_at     = now + LED_DURATION_US;
                        gpio_set_level(GPIO_NUM_2, 1); // LED ON
                    }
                } 
                // Test Case 3: Model thinks it's "Hey Pluto", but confidence is too low
                else {
                    // We only log this occasionally to avoid spam, or let the main [OUT] log handle it
                    // The [OUT] log at the bottom will show the exact percentage.
                }
            } 
            // Test Case 4: Model is confident it's NOT the wake word (Unknown class)
            else {
                // Handled by the standard [OUT] log below
            }

            int64_t cooldown_left_ms = in_cooldown ? (cooldown_until - now) / 1000LL : 0LL;
            ESP_LOGI(TAG, "[OUT] Wake=%.0f%% Unknown=%.0f%% | cooldown=%lldms",
                     prob0 * 100.0f, prob1 * 100.0f, cooldown_left_ms);

        }
    }
}

void inference_engine_init() {
    init_dct();

    const tflite::Model *model = wake_word_model_get();

    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        abort();
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Read input quantization params FROM THIS MODEL — never hardcode these
    g_input_scale = input_tensor->params.scale;
    g_input_zero  = input_tensor->params.zero_point;

    ESP_LOGI(TAG, "TFLite init OK — Arena: %d bytes | Input quant: scale=%.6f zero=%d",
             interpreter->arena_used_bytes(), g_input_scale, g_input_zero);
    ESP_LOGI(TAG, "Output quant: scale=%.6f zero=%d",
             output_tensor->params.scale, output_tensor->params.zero_point);

    xTaskCreatePinnedToCore(
        inference_task,
        "Inference",
        16384, // Slightly larger stack to hold the local_ring buffer securely
        NULL,
        configMAX_PRIORITIES - 3,
        &inference_task_handle, // Store handle for notifications
        1 // Core 1
    );
}
