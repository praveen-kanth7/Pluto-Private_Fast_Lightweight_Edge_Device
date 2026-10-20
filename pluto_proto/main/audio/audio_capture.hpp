#pragma once

#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#define AUDIO_FRAME_SAMPLES 16000
#define AUDIO_SAMPLE_RATE   16000

struct AudioFrame {
    int16_t samples[AUDIO_FRAME_SAMPLES];
};

// Initializes the I2S hardware and spawns the AudioCaptureTask
// pcm_queue: Queue to push filled AudioFrame* into
// free_queue: Queue to pop available AudioFrame* from
void audio_capture_init(QueueHandle_t pcm_queue, QueueHandle_t free_queue);
