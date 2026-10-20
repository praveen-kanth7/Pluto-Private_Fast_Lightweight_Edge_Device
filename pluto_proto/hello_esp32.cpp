#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_log.h"

#include "driver/i2s_std.h"

#include "mfcc.h"

static const char *TAG = "INMP441";
float mfcc_features[MFCC_COEFFS][MFCC_FRAMES];

// ==============================
// INMP441 → ESP32-S3
// ==============================
#define I2S_PORT        I2S_NUM_0

#define I2S_BCLK        GPIO_NUM_4
#define I2S_WS          GPIO_NUM_5
#define I2S_DIN         GPIO_NUM_6

// ==============================
// AUDIO CONFIG
// ==============================
#define SAMPLE_RATE     16000
#define NUM_SAMPLES     16000

static i2s_chan_handle_t rx_handle;

// Store 32-bit I2S samples
static int32_t raw_samples[NUM_SAMPLES];

// Store converted audio
static int16_t pcm_samples[NUM_SAMPLES];


// ============================================================
// I2S INITIALIZATION
// ============================================================
static void i2s_init()
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(
        i2s_new_channel(&chan_cfg, NULL, &rx_handle)
    );

    i2s_std_config_t std_cfg = {

        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK,
            .ws   = I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false
            }
        }
    };

    // INMP441 sends on LEFT channel when L/R = GND
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(rx_handle, &std_cfg)
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_handle)
    );

    ESP_LOGI(TAG, "I2S initialized");
}


// ============================================================
// CAPTURE 1 SECOND
// ============================================================
static void capture_audio()
{
    ESP_LOGI(TAG, "Capturing 1 second...");

    size_t bytes_read = 0;

    int samples_received = 0;

    while (samples_received < NUM_SAMPLES)
    {
        size_t bytes_to_read =
            (NUM_SAMPLES - samples_received) * sizeof(int32_t);

        esp_err_t ret =
            i2s_channel_read(
                rx_handle,
                raw_samples + samples_received,
                bytes_to_read,
                &bytes_read,
                portMAX_DELAY
            );

        if (ret != ESP_OK)
        {
            ESP_LOGE(TAG, "I2S read failed: %s",
                     esp_err_to_name(ret));
            continue;
        }

        int new_samples =
            bytes_read / sizeof(int32_t);

        samples_received += new_samples;
    }

    ESP_LOGI(TAG,
             "Captured %d samples",
             samples_received);
}


// ============================================================
// CONVERT INMP441 24-BIT → INT16
// ============================================================
static void convert_audio()
{
    int32_t min24 = INT32_MAX;
    int32_t max24 = INT32_MIN;

    int16_t min16 = INT16_MAX;
    int16_t max16 = INT16_MIN;

    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        // Extract signed 24-bit sample
        int32_t sample24 = raw_samples[i] >> 8;

        if (sample24 < min24)
            min24 = sample24;

        if (sample24 > max24)
            max24 = sample24;

        // 24-bit → 16-bit
        int32_t sample16 = sample24 >> 8;

        if (sample16 > 32767)
            sample16 = 32767;

        if (sample16 < -32768)
            sample16 = -32768;

        pcm_samples[i] = (int16_t)sample16;

        if (pcm_samples[i] < min16)
            min16 = pcm_samples[i];

        if (pcm_samples[i] > max16)
            max16 = pcm_samples[i];

        sum += pcm_samples[i];

        sum_sq +=
            (double)pcm_samples[i] *
            (double)pcm_samples[i];
    }

    double mean =
        sum / NUM_SAMPLES;

    double rms =
        sqrt(sum_sq / NUM_SAMPLES);

    ESP_LOGI(TAG,
             "24-bit min = %" PRId32,
             min24);

    ESP_LOGI(TAG,
             "24-bit max = %" PRId32,
             max24);

    ESP_LOGI(TAG,
             "PCM16 min = %d",
             min16);

    ESP_LOGI(TAG,
             "PCM16 max = %d",
             max16);

    ESP_LOGI(TAG,
             "PCM16 mean = %.2f",
             mean);

    ESP_LOGI(TAG,
             "PCM16 RMS = %.2f",
             rms);

    
}


// ============================================================
// PRINT FIRST SAMPLES
// ============================================================
static void print_samples()
{
    ESP_LOGI(TAG, "First 40 PCM samples:");

    for (int i = 0; i < 40; i++)
    {
        printf(
            "%d%s",
            pcm_samples[i],
            (i == 39) ? "\n" : ", "
        );
    }
}

// ============================================================
// PRINT MFCC INFORMATION
// ============================================================



// ============================================================
// MAIN TASK
// ============================================================
extern "C" void app_main()
{
    ESP_LOGI(TAG,
             "======================================");

    ESP_LOGI(TAG,
             "ESP32-S3 INMP441 AUDIO TEST");

    ESP_LOGI(TAG,
             "======================================");

    i2s_init();

    vTaskDelay(
        pdMS_TO_TICKS(500)
    );

    while (true)
    {
        capture_audio();

        convert_audio();

        print_samples();

        // ==========================================
        // MFCC
        // ==========================================
        ESP_LOGI(TAG, "Computing MFCC...");

        compute_mfcc(
            pcm_samples,
            mfcc_features
        );

        print_mfcc(mfcc_features);

        ESP_LOGI(TAG,
                "======================================");

        vTaskDelay(
            pdMS_TO_TICKS(2000)
        );
    }
}