#include "audio_capture.hpp"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>

static const char *TAG = "AudioCapture";

#define I2S_NUM       I2S_NUM_0
#define I2S_BCLK      GPIO_NUM_4
#define I2S_WS        GPIO_NUM_5
#define I2S_SD        GPIO_NUM_6

static i2s_chan_handle_t rx_handle = NULL;
static QueueHandle_t out_queue;
static QueueHandle_t in_queue;

static void audio_capture_task(void *arg) {
    ESP_LOGI(TAG, "AudioCaptureTask started on Core %d", xPortGetCoreID());

    while (1) {
        AudioFrame *frame = nullptr;
        // Wait for a free buffer
        if (xQueueReceive(in_queue, &frame, portMAX_DELAY) == pdTRUE) {
            
            // Read exactly AUDIO_FRAME_SAMPLES
            // INMP441 gives 32-bit words for 16kHz
            size_t bytes_to_read = AUDIO_FRAME_SAMPLES * sizeof(int32_t);
            size_t bytes_read = 0;
            
            // Read in chunks to avoid allocating 64KB on stack
            const int chunk_size_samples = 250; 
            int32_t raw_buffer[chunk_size_samples];
            
            int samples_received = 0;
            int64_t start_time = esp_timer_get_time();

            while (samples_received < AUDIO_FRAME_SAMPLES) {
                int to_read = (AUDIO_FRAME_SAMPLES - samples_received);
                if (to_read > chunk_size_samples) to_read = chunk_size_samples;

                size_t chunk_bytes_read = 0;
                esp_err_t ret = i2s_channel_read(
                    rx_handle,
                    raw_buffer,
                    to_read * sizeof(int32_t),
                    &chunk_bytes_read,
                    portMAX_DELAY
                );

                if (ret == ESP_OK) {
                    int samples_read = chunk_bytes_read / sizeof(int32_t);
                    for (int i = 0; i < samples_read; i++) {
                        frame->samples[samples_received + i] = (int16_t)(raw_buffer[i] >> 16);
                    }
                    samples_received += samples_read;
                } else {
                    ESP_LOGE(TAG, "I2S read failed: %d", ret);
                }
            }

            int64_t end_time = esp_timer_get_time();
            ESP_LOGI(TAG, "Captured 1s audio in %lld us | HighWaterMark: %d words", 
                     end_time - start_time, uxTaskGetStackHighWaterMark(NULL));
            
            // Send to processing
            if (xQueueSend(out_queue, &frame, 0) != pdTRUE) {
                ESP_LOGE(TAG, "DSP queue full! Dropping frame.");
                // Return to free queue
                xQueueSend(in_queue, &frame, 0);
            }
        }
    }
}

void audio_capture_init(QueueHandle_t pcm_queue, QueueHandle_t free_queue) {
    out_queue = pcm_queue;
    in_queue = free_queue;

    ESP_LOGI(TAG, "Initializing INMP441...");

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
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

    // INMP441 L/R connected to GND -> LEFT channel
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    ESP_LOGI(TAG, "INMP441 initialized");

    // Start audio capture task on Core 0
    xTaskCreatePinnedToCore(
        audio_capture_task,
        "AudioCapture",
        4096,
        NULL,
        configMAX_PRIORITIES - 1,
        NULL,
        0 // Core 0
    );
}
