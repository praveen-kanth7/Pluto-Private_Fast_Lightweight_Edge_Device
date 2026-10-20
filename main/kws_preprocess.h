// ============================================================
// ESP32-S3 + INMP441 + TFLite Micro
// HEY PLUTO Wake Word Detection
//
// INMP441:
//   SCK/BCLK -> GPIO 4
//   WS       -> GPIO 5
//   SD       -> GPIO 6
//   L/R      -> GND
//   VDD      -> 3.3V
//   GND      -> GND
//
// Model:
//   ds_cnn_int8.tflite
//
// Input:
//   [1, 40, 49, 1]
//   int8
//   scale = 0.0480119027197361
//   zero_point = 49
//
// Output:
//   [1, 3]
//   int8
//   scale = 0.00390625
//   zero_point = -128
//
// Classes:
//   0 = WAKE WORD
//   1 = UNKNOWN
//   2 = NOISE
//
// MFCC training parameters:
//   sample rate = 16000 Hz
//   n_mfcc      = 40
//   n_fft       = 480
//   hop_length  = 320
//   n_mels      = 40
//   center      = false
//
// Training normalization:
//   mean = -5.2158537
//   std  = 46.219826
// ============================================================

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "kws_preprocess.h"
#include "test_audio.h"

// Set to 1 only for the deterministic fixture run. It never reads I2S and
// emits machine-readable MFCC/INT8 bytes for tools/validate_preprocess.py.
#define KWS_DETERMINISTIC_VALIDATION 0

// TensorFlow Lite Micro
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================
// MODEL
// ============================================================

extern const uint8_t ds_cnn_int8_tflite_start[]
    asm("_binary_ds_cnn_int8_tflite_start");

extern const uint8_t ds_cnn_int8_tflite_end[]
    asm("_binary_ds_cnn_int8_tflite_end");

// ============================================================
// TAG
// ============================================================

static const char *TAG = "PLUTO";

// ============================================================
// I2S
// ============================================================

#define I2S_NUM       I2S_NUM_0

#define I2S_BCLK      GPIO_NUM_4
#define I2S_WS        GPIO_NUM_5
#define I2S_SD        GPIO_NUM_6

#define SAMPLE_RATE   16000

// ============================================================
// AUDIO
// ============================================================

#define AUDIO_SAMPLES 16000

static int16_t audio_buffer[AUDIO_SAMPLES];

// ============================================================
// MFCC
// ============================================================

#define NUM_MFCC      40
#define NUM_FRAMES    49

#define FFT_SIZE      480
#define HOP_LENGTH    320
#define NUM_MELS      40

// ============================================================
// TRAINING NORMALIZATION
// ============================================================

#define TRAIN_MEAN    (-5.2158537f)
#define TRAIN_STD     (46.219826f)

// ============================================================
// INT8 QUANTIZATION
// ============================================================

#define INPUT_SCALE   (0.0480119027197361f)
#define INPUT_ZERO    (49)

// ============================================================
// I2S HANDLE
// ============================================================

static i2s_chan_handle_t rx_handle = NULL;

// ============================================================
// TFLITE
// ============================================================

static tflite::MicroInterpreter *interpreter = nullptr;

static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

#if KWS_DETERMINISTIC_VALIDATION
static float validation_mfcc[KWS_INPUT_ELEMENTS];
#endif

// ============================================================
// TENSOR ARENA
// ============================================================

#define TENSOR_ARENA_SIZE (120 * 1024)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// ============================================================
// FFT WORKING ARRAYS
// ============================================================

static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];

static float power_spectrum[FFT_SIZE / 2 + 1];

static float mel_energies[NUM_MELS];

static float mfcc[NUM_MFCC][NUM_FRAMES];

// ============================================================
// HANN WINDOW
// ============================================================

static float hann_window[FFT_SIZE];

// ============================================================
// MEL FILTER BANK
// ============================================================

static int mel_start[NUM_MELS];
static int mel_end[NUM_MELS];

static float mel_filter[NUM_MELS][FFT_SIZE / 2 + 1];

// ============================================================
// PI
// ============================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// I2S INITIALIZATION
// ============================================================

static void init_i2s()
{
    ESP_LOGI(TAG, "Initializing INMP441...");

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM,
            I2S_ROLE_MASTER
        );

    ESP_ERROR_CHECK(
        i2s_new_channel(
            &chan_cfg,
            NULL,
            &rx_handle
        )
    );

    i2s_std_config_t std_cfg = {

        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(
                SAMPLE_RATE
            ),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

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

    // INMP441 L/R connected to GND
    // Therefore LEFT channel.
    std_cfg.slot_cfg.slot_mask =
        I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg
        )
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(
            rx_handle
        )
    );

    ESP_LOGI(TAG, "INMP441 initialized");
    ESP_LOGI(TAG, "BCLK = GPIO %d", I2S_BCLK);
    ESP_LOGI(TAG, "WS   = GPIO %d", I2S_WS);
    ESP_LOGI(TAG, "SD   = GPIO %d", I2S_SD);
}

// ============================================================
// READ ONE PCM SAMPLE
//
// INMP441 provides 24-bit data inside 32-bit I2S word.
//
// Based on our diagnostic test:
//
// raw >> 16 gives the correct signed PCM16 representation.
//
// Then normalize exactly like the training data:
// PCM16 / 32768.0
// ============================================================

static inline int16_t read_pcm16()
{
    int32_t raw = 0;

    size_t bytes_read = 0;

    ESP_ERROR_CHECK(
        i2s_channel_read(
            rx_handle,
            &raw,
            sizeof(raw),
            &bytes_read,
            portMAX_DELAY
        )
    );

    return (int16_t)(raw >> 16);
}

// ============================================================
// CAPTURE 1 SECOND
// ============================================================

static void capture_audio()
{
    ESP_LOGI(TAG, "Capturing 1 second...");

    for (int i = 0; i < AUDIO_SAMPLES; i++)
    {
        audio_buffer[i] =
            read_pcm16();
    }

    ESP_LOGI(TAG, "Audio capture complete");
}

// ============================================================
// HANN WINDOW
// ============================================================

static void init_hann_window()
{
    for (int n = 0; n < FFT_SIZE; n++)
    {
        hann_window[n] =
            0.5f *
            (1.0f -
             cosf(
                 2.0f *
                 M_PI *
                 n /
                 FFT_SIZE
             ));
    }
}

// ============================================================
// FFT
//
// Iterative radix-2 FFT.
//
// NOTE:
// FFT_SIZE = 480 is NOT power of 2.
// Therefore we use a direct DFT here for correctness.
// This is slower but keeps the implementation simple and
// makes debugging easier.
//
// Once the pipeline is verified, this can be optimized.
// ============================================================

static void compute_power_spectrum(
    const float *input
)
{
    for (int k = 0; k <= FFT_SIZE / 2; k++)
    {
        float real = 0.0f;
        float imag = 0.0f;

        for (int n = 0; n < FFT_SIZE; n++)
        {
            float angle =
                -2.0f *
                M_PI *
                k *
                n /
                FFT_SIZE;

            float sample =
                input[n] *
                hann_window[n];

            real +=
                sample *
                cosf(angle);

            imag +=
                sample *
                sinf(angle);
        }

        power_spectrum[k] =
            real * real +
            imag * imag;
    }
}

// ============================================================
// MEL SCALE
//
// Librosa uses:
//   Slaney mel scale
//
// mel = 1127 * ln(1 + f/700)
// ============================================================

static float hz_to_mel(float hz)
{
    return 1127.0f *
           logf(
               1.0f +
               hz / 700.0f
           );
}

static float mel_to_hz(float mel)
{
    return 700.0f *
           (expf(mel / 1127.0f) - 1.0f);
}

// ============================================================
// INITIALIZE MEL FILTER BANK
// ============================================================

static void init_mel_filterbank()
{
    const float f_min = 0.0f;

    const float f_max =
        SAMPLE_RATE / 2.0f;

    const float mel_min =
        hz_to_mel(f_min);

    const float mel_max =
        hz_to_mel(f_max);

    float mel_points[NUM_MELS + 2];

    int bins[NUM_MELS + 2];

    for (int i = 0; i < NUM_MELS + 2; i++)
    {
        mel_points[i] =
            mel_min +
            (mel_max - mel_min) *
            i /
            (NUM_MELS + 1);

        float hz =
            mel_to_hz(
                mel_points[i]
            );

        bins[i] =
            (int)floorf(
                (FFT_SIZE + 1) *
                hz /
                SAMPLE_RATE
            );
    }

    memset(
        mel_filter,
        0,
        sizeof(mel_filter)
    );

    for (int m = 0; m < NUM_MELS; m++)
    {
        int left =
            bins[m];

        int center =
            bins[m + 1];

        int right =
            bins[m + 2];

        mel_start[m] = left;
        mel_end[m] = right;

        // Rising slope
        for (
            int k = left;
            k < center;
            k++
        )
        {
            if (
                k >= 0 &&
                k <= FFT_SIZE / 2 &&
                center != left
            )
            {
                mel_filter[m][k] =
                    (float)(k - left) /
                    (float)(center - left);
            }
        }

        // Falling slope
        for (
            int k = center;
            k < right;
            k++
        )
        {
            if (
                k >= 0 &&
                k <= FFT_SIZE / 2 &&
                right != center
            )
            {
                mel_filter[m][k] =
                    (float)(right - k) /
                    (float)(right - center);
            }
        }
    }
}

// ============================================================
// DCT-II
//
// Librosa:
// scipy.fftpack.dct(..., type=2, norm='ortho')
// ============================================================

static float dct_basis(
    int k,
    int n
)
{
    return cosf(
        M_PI *
        k *
        (2.0f * n + 1.0f) /
        (2.0f * NUM_MELS)
    );
}

// ============================================================
// COMPUTE MFCC
//
// Output layout:
//
// mfcc[mfcc_index][frame_index]
//
// Later copied to TFLite as:
//
// frame-major:
// mfcc[m][f]
//
// flattened:
// m * NUM_FRAMES + f
// ============================================================

static void compute_mfcc()
{
    ESP_LOGI(TAG, "Computing MFCC...");

    float frame[FFT_SIZE];

    float mel_db[NUM_MELS][NUM_FRAMES];

    float global_max_db = -100000.0f;

    // --------------------------------------------------------
    // First calculate all mel energies
    // --------------------------------------------------------

    for (int frame_idx = 0;
         frame_idx < NUM_FRAMES;
         frame_idx++)
    {
        int start =
            frame_idx *
            HOP_LENGTH;

        // --------------------------------------------
        // Convert PCM16 to float
        // EXACT training normalization
        // --------------------------------------------

        for (int n = 0; n < FFT_SIZE; n++)
        {
            int index =
                start + n;

            if (index < AUDIO_SAMPLES)
            {
                frame[n] =
                    (float)
                        audio_buffer[index]
                    /
                    32768.0f;
            }
            else
            {
                frame[n] = 0.0f;
            }
        }

        // --------------------------------------------
        // Power spectrum
        // --------------------------------------------

        compute_power_spectrum(frame);

        // --------------------------------------------
        // Mel filterbank
        // --------------------------------------------

        for (int m = 0;
             m < NUM_MELS;
             m++)
        {
            float energy = 0.0f;

            int start_bin =
                mel_start[m];

            int end_bin =
                mel_end[m];

            if (start_bin < 0)
                start_bin = 0;

            if (end_bin >
                FFT_SIZE / 2)
            {
                end_bin =
                    FFT_SIZE / 2;
            }

            for (
                int k = start_bin;
                k <= end_bin;
                k++
            )
            {
                energy +=
                    power_spectrum[k] *
                    mel_filter[m][k];
            }

            // Avoid log(0)
            if (energy < 1e-10f)
                energy = 1e-10f;

            // ----------------------------------------
            // Convert power to dB
            //
            // Temporarily store log power.
            // Global max normalization is applied
            // after all frames are calculated.
            // ----------------------------------------

            float db =
                10.0f *
                log10f(energy);

            mel_db[m][frame_idx] =
                db;

            if (db > global_max_db)
            {
                global_max_db = db;
            }
        }
    }

    // --------------------------------------------------------
    // Librosa power_to_db(ref=np.max, top_db=80)
    // --------------------------------------------------------

    float floor_db =
        global_max_db - 80.0f;

    for (int m = 0;
         m < NUM_MELS;
         m++)
    {
        for (int f = 0;
             f < NUM_FRAMES;
             f++)
        {
            float db =
                mel_db[m][f];

            if (db < floor_db)
                db = floor_db;

            // Relative to global maximum
            mel_db[m][f] =
                db - global_max_db;
        }
    }

    // --------------------------------------------------------
    // DCT
    // --------------------------------------------------------

    const float dct_scale =
        sqrtf(
            2.0f /
            NUM_MELS
        );

    for (int k = 0;
         k < NUM_MFCC;
         k++)
    {
        for (int f = 0;
             f < NUM_FRAMES;
             f++)
        {
            float sum = 0.0f;

            for (int n = 0;
                 n < NUM_MELS;
                 n++)
            {
                sum +=
                    mel_db[n][f] *
                    dct_basis(k, n);
            }

            float scale =
                dct_scale;

            // DCT orthonormal correction
            if (k == 0)
            {
                scale =
                    sqrtf(
                        1.0f /
                        NUM_MELS
                    );
            }

            mfcc[k][f] =
                sum * scale;
        }
    }

    ESP_LOGI(
        TAG,
        "MFCC complete. Global max dB = %.3f",
        global_max_db
    );
}

// ============================================================
// LOAD MFCC INTO TFLITE INPUT
// ============================================================

static void prepare_input()
{
    ESP_LOGI(TAG, "Preparing INT8 input...");

    int min_value = 127;
    int max_value = -128;

    float sum = 0.0f;

    const int total =
        NUM_MFCC *
        NUM_FRAMES;

    for (int m = 0;
         m < NUM_MFCC;
         m++)
    {
        for (int f = 0;
             f < NUM_FRAMES;
             f++)
        {
            // --------------------------------------------
            // Training normalization
            // --------------------------------------------

            float normalized =
                (
                    mfcc[m][f] -
                    TRAIN_MEAN
                )
                /
                TRAIN_STD;

            // --------------------------------------------
            // INT8 quantization
            // --------------------------------------------

            int q =
                (int)roundf(
                    normalized /
                    INPUT_SCALE
                )
                +
                INPUT_ZERO;

            // Clamp
            if (q < -128)
                q = -128;