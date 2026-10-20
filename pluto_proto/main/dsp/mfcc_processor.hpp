#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include <stdint.h>
#include "../audio/audio_capture.hpp"

#define MFCC_COEFFS 40
#define MFCC_FRAMES 49
#define MFCC_TENSOR_SIZE (MFCC_COEFFS * MFCC_FRAMES)

struct FeatureTensor {
    int8_t data[MFCC_TENSOR_SIZE];
};

// Initializes DSP modules and spawns the DSPTask
// pcm_queue: Queue to pop filled AudioFrame* from
// pcm_free_queue: Queue to push processed AudioFrame* back to
// feature_queue: Queue to push filled FeatureTensor* into
// feature_free_queue: Queue to pop available FeatureTensor* from
void dsp_processor_init(QueueHandle_t pcm_queue, QueueHandle_t pcm_free_queue,
                        QueueHandle_t feature_queue, QueueHandle_t feature_free_queue);
