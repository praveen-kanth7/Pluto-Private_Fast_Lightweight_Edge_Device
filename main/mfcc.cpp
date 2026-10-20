#include "mfcc.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#include "esp_log.h"

static const char *TAG = "MFCC";

#define SAMPLE_RATE 16000.0f
#define PI 3.14159265358979323846f

#define EPSILON 1e-10f
#define TOP_DB 80.0f

// ============================================================
// TRAINING PARAMETERS
// ============================================================

static constexpr float TRAIN_MEAN = -5.2158537f;
static constexpr float TRAIN_STD  = 46.219826f;

// ============================================================
// STATIC DSP BUFFERS
// ============================================================

static float window_func[FRAME_SIZE];

static float fft_input[FFT_SIZE];
static float fft_real[FFT_SIZE];
static float fft_imag[FFT_SIZE];

static float stage_real[FFT_SIZE];
static float stage_imag[FFT_SIZE];

static float power_spectrum[FFT_SIZE / 2 + 1];

static float mel_bank[MEL_FILTERS][FFT_SIZE / 2 + 1];

static float mel_db[MEL_FILTERS][MFCC_FRAMES];

static float dct_table[MFCC_COEFFS][MEL_FILTERS];

static bool initialized = false;

// ============================================================
// FAST 480-POINT FFT TABLES
//
// 480 = 32 x 15
// ============================================================

static float w32r[16];
static float w32i[16];

static float w15r[15][15];
static float w15i[15][15];

static float w480r[32][15];
static float w480i[32][15];

// ============================================================
// HANN WINDOW
//
// Librosa uses a periodic Hann window for STFT.
// ============================================================

static void init_window()
{
    for (int n = 0; n < FRAME_SIZE; ++n)
    {
        window_func[n] =
            0.5f -
            0.5f *
            cosf(
                2.0f *
                PI *
                (float)n /
                (float)FRAME_SIZE
            );
    }
}

// ============================================================
// FAST FFT TABLE INITIALIZATION
// ============================================================

static void init_fft_tables()
{
    // 32-point FFT twiddles
    for (int k = 0; k < 16; ++k)
    {
        float a =
            -2.0f *
            PI *
            (float)k /
            32.0f;

        w32r[k] = cosf(a);
        w32i[k] = sinf(a);
    }

    // 15-point DFT twiddles
    for (int k = 0; k < 15; ++k)
    {
        for (int n = 0; n < 15; ++n)
        {
            float a =
                -2.0f *
                PI *
                (float)k *
                (float)n /
                15.0f;

            w15r[k][n] = cosf(a);
            w15i[k][n] = sinf(a);
        }
    }

    // 480-point twiddle factors
    for (int k2 = 0; k2 < 32; ++k2)
    {
        for (int n1 = 0; n1 < 15; ++n1)
        {
            float a =
                -2.0f *
                PI *
                (float)k2 *
                (float)n1 /
                480.0f;

            w480r[k2][n1] = cosf(a);
            w480i[k2][n1] = sinf(a);
        }
    }
}

// ============================================================
// 32-POINT RADIX-2 FFT
// ============================================================

static void fft32(
    float *real,
    float *imag
)
{
    // Bit reversal
    int j = 0;

    for (int i = 1; i < 32; ++i)
    {
        int bit = 16;

        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j)
        {
            float t;

            t = real[i];
            real[i] = real[j];
            real[j] = t;

            t = imag[i];
            imag[i] = imag[j];
            imag[j] = t;
        }
    }

    // FFT stages
    for (int len = 2; len <= 32; len <<= 1)
    {
        int half = len >> 1;

        for (int base = 0;
             base < 32;
             base += len)
        {
            for (int jj = 0;
                 jj < half;
                 ++jj)
            {
                int table_index =
                    jj * 16 / half;

                float wr =
                    w32r[table_index];

                float wi =
                    w32i[table_index];

                float ar =
                    real[base + jj];

                float ai =
                    imag[base + jj];

                float br =
                    real[base + jj + half];

                float bi =
                    imag[base + jj + half];

                float tr =
                    br * wr -
                    bi * wi;

                float ti =
                    br * wi +
                    bi * wr;

                real[base + jj] =
                    ar + tr;

                imag[base + jj] =
                    ai + ti;

                real[base + jj + half] =
                    ar - tr;

                imag[base + jj + half] =
                    ai - ti;
            }
        }
    }
}

// ============================================================
// EXACT 480-POINT FFT
//
// 480 = 32 x 15
//
// n = n1 + 15*n2
// k = k2 + 32*k1
// ============================================================

static void fft480()
{
    // --------------------------------------------------------
    // First stage: 15 separate 32-point FFTs
    // --------------------------------------------------------

    for (int n1 = 0;
         n1 < 15;
         ++n1)
    {
        for (int n2 = 0;
             n2 < 32;
             ++n2)
        {
            int index =
                n1 +
                15 * n2;

            fft_real[n2] =
                fft_input[index];

            fft_imag[n2] =
                0.0f;
        }

        fft32(
            fft_real,
            fft_imag
        );

        for (int k2 = 0;
             k2 < 32;
             ++k2)
        {
            stage_real[
                n1 + 15 * k2
            ] =
                fft_real[k2];

            stage_imag[
                n1 + 15 * k2
            ] =
                fft_imag[k2];
        }
    }

    // --------------------------------------------------------
    // Second stage
    // --------------------------------------------------------

    for (int k2 = 0;
         k2 < 32;
         ++k2)
    {
        for (int k1 = 0;
             k1 < 15;
             ++k1)
        {
            float sum_real = 0.0f;
            float sum_imag = 0.0f;

            for (int n1 = 0;
                 n1 < 15;
                 ++n1)
            {
                float ar =
                    stage_real[
                        n1 + 15 * k2
                    ];

                float ai =
                    stage_imag[
                        n1 + 15 * k2
                    ];

                float wr1 =
                    w15r[k1][n1];

                float wi1 =
                    w15i[k1][n1];

                float wr2 =
                    w480r[k2][n1];

                float wi2 =
                    w480i[k2][n1];

                // Multiply by 15-point twiddle
                float br =
                    ar * wr1 -
                    ai * wi1;

                float bi =
                    ar * wi1 +
                    ai * wr1;

                // Multiply by 480-point twiddle
                sum_real +=
                    br * wr2 -
                    bi * wi2;

                sum_imag +=
                    br * wi2 +
                    bi * wr2;
            }

            int output_index =
                k2 +
                32 * k1;

            fft_real[output_index] =
                sum_real;

            fft_imag[output_index] =
                sum_imag;
        }
    }
}

// ============================================================
// SLANEY MEL SCALE
//
// This matches librosa.filters.mel() default:
// htk=False
// ============================================================

static float hz_to_mel(float hz)
{
    const float f_sp =
        200.0f / 3.0f;

    const float min_hz =
        1000.0f;

    const float min_mel =
        min_hz / f_sp;

    const float logstep =
        logf(6.4f) / 27.0f;

    if (hz < min_hz)
    {
        return hz / f_sp;
    }

    return min_mel +
           logf(hz / min_hz) /
           logstep;
}

static float mel_to_hz(float mel)
{
    const float f_sp =
        200.0f / 3.0f;

    const float min_hz =
        1000.0f;

    const float min_mel =
        min_hz / f_sp;

    const float logstep =
        logf(6.4f) / 27.0f;

    if (mel < min_mel)
    {
        return mel * f_sp;
    }

    return min_hz *
           expf(
               logstep *
               (mel - min_mel)
           );
}

// ============================================================
// MEL FILTER BANK
//
// Librosa defaults:
//
// n_mels = 40
// fmin   = 0
// fmax   = 8000
// htk    = false
// norm   = "slaney"
// ============================================================

static void init_mel_filterbank()
{
    memset(
        mel_bank,
        0,
        sizeof(mel_bank)
    );

    float mel_min =
        hz_to_mel(0.0f);

    float mel_max =
        hz_to_mel(
            SAMPLE_RATE / 2.0f
        );

    float mel_points[
        MEL_FILTERS + 2
    ];

    float hz_points[
        MEL_FILTERS + 2
    ];

    for (int i = 0;
         i < MEL_FILTERS + 2;
         ++i)
    {
        mel_points[i] =
            mel_min +
            (mel_max - mel_min) *
            (float)i /
            (float)(MEL_FILTERS + 1);

        hz_points[i] =
            mel_to_hz(
                mel_points[i]
            );
    }

    for (int m = 0;
         m < MEL_FILTERS;
         ++m)
    {
        float lower =
            hz_points[m];

        float center =
            hz_points[m + 1];

        float upper =
            hz_points[m + 2];

        // Slaney normalization
        float norm =
            2.0f /
            (upper - lower);

        for (int k = 0;
             k <= FFT_SIZE / 2;
             ++k)
        {
            float frequency =
                (float)k *
                SAMPLE_RATE /
                (float)FFT_SIZE;

            float weight = 0.0f;

            if (frequency > lower &&
                frequency < center)
            {
                weight =
                    (frequency - lower) /
                    (center - lower);
            }
            else if (frequency >= center &&
                     frequency < upper)
            {
                weight =
                    (upper - frequency) /
                    (upper - center);
            }

            if (weight > 0.0f)
            {
                weight *= norm;
            }

            mel_bank[m][k] =
                weight;
        }
    }
}

// ============================================================
// DCT-II ORTHONORMAL TABLE
//
// Matches scipy/librosa:
// type=2
// norm="ortho"
// ============================================================

static void init_dct()
{
    const float scale0 =
        sqrtf(
            1.0f /
            (float)MEL_FILTERS
        );

    const float scale =
        sqrtf(
            2.0f /
            (float)MEL_FILTERS
        );

    for (int c = 0;
         c < MFCC_COEFFS;
         ++c)
    {
        for (int m = 0;
             m < MEL_FILTERS;
             ++m)
        {
            float scale_c =
                (c == 0)
                    ? scale0
                    : scale;

            dct_table[c][m] =
                scale_c *
                cosf(
                    PI *
                    (float)c *
                    ((float)m + 0.5f) /
                    (float)MEL_FILTERS
                );
        }
    }
}

// ============================================================
// INITIALIZATION
// ============================================================

static void mfcc_init()
{
    if (initialized)
    {
        return;
    }

    ESP_LOGI(
        TAG,
        "Initializing exact Librosa-compatible MFCC"
    );

    init_window();

    init_fft_tables();

    init_mel_filterbank();

    init_dct();

    initialized = true;

    ESP_LOGI(
        TAG,
        "MFCC initialized: FFT=%d FRAME=%d HOP=%d MEL=%d MFCC=%d",
        FFT_SIZE,
        FRAME_SIZE,
        FRAME_HOP,
        MEL_FILTERS,
        MFCC_COEFFS
    );
}

// ============================================================
// COMPUTE MFCC
//
// Input:
//   16000 PCM16 samples
//
// Output:
//   [40][49]
//
// Pipeline:
//
// PCM16
//   ↓
// float / 32768
//   ↓
// Hann
//   ↓
// 480 FFT
//   ↓
// power spectrum
//   ↓
// Slaney mel
//   ↓
// power_to_db(ref=max, top_db=80)
//   ↓
// DCT-II ortho
// ============================================================

void compute_mfcc(
    const int16_t *audio,
    float mfcc[MFCC_COEFFS][MFCC_FRAMES]
)
{
    mfcc_init();

    // --------------------------------------------------------
    // 49 STFT frames
    // --------------------------------------------------------

    for (int frame = 0;
         frame < MFCC_FRAMES;
         ++frame)
    {
        int start =
            frame * FRAME_HOP;

        // ----------------------------------------------------
        // PCM16 → float + Hann
        // ----------------------------------------------------

        for (int n = 0;
             n < FRAME_SIZE;
             ++n)
        {
            float sample =
                (float)audio[start + n] /
                32768.0f;

            fft_input[n] =
                sample *
                window_func[n];
        }

        // ----------------------------------------------------
        // 480-point FFT
        // ----------------------------------------------------

        fft480();

        // ----------------------------------------------------
        // Power spectrum
        //
        // Librosa STFT does NOT divide FFT by N.
        // ----------------------------------------------------

        for (int k = 0;
             k <= FFT_SIZE / 2;
             ++k)
        {
            float re =
                fft_real[k];

            float im =
                fft_imag[k];

            power_spectrum[k] =
                re * re +
                im * im;
        }

        // ----------------------------------------------------
        // Mel filter bank
        // ----------------------------------------------------

        for (int m = 0;
             m < MEL_FILTERS;
             ++m)
        {
            float energy = 0.0f;

            for (int k = 0;
                 k <= FFT_SIZE / 2;
                 ++k)
            {
                energy +=
                    power_spectrum[k] *
                    mel_bank[m][k];
            }

            if (energy < EPSILON)
            {
                energy = EPSILON;
            }

            // Absolute dB for now.
            //
            // We subtract the global maximum below,
            // exactly like librosa.power_to_db(ref=np.max).

            mel_db[m][frame] =
                10.0f *
                log10f(energy);
        }
    }

    // --------------------------------------------------------
    // Find global maximum
    //
    // librosa.power_to_db(ref=np.max)
    //
    // Result maximum becomes 0 dB.
    // --------------------------------------------------------

    float global_max =
        -INFINITY;

    for (int m = 0;
         m < MEL_FILTERS;
         ++m)
    {
        for (int frame = 0;
             frame < MFCC_FRAMES;
             ++frame)
        {
            if (mel_db[m][frame] >
                global_max)
            {
                global_max =
                    mel_db[m][frame];
            }
        }
    }

    float floor_db =
        global_max -
        TOP_DB;

    // --------------------------------------------------------
    // Normalize to reference maximum
    // --------------------------------------------------------

    for (int m = 0;
         m < MEL_FILTERS;
         ++m)
    {
        for (int frame = 0;
             frame < MFCC_FRAMES;
             ++frame)
        {
            float db =
                mel_db[m][frame] -
                global_max;

            if (db < -TOP_DB)
            {
                db = -TOP_DB;
            }

            mel_db[m][frame] =
                db;
        }
    }

    // --------------------------------------------------------
    // DCT-II
    // --------------------------------------------------------

    for (int c = 0;
         c < MFCC_COEFFS;
         ++c)
    {