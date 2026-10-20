#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "../dsp/mfcc_processor.hpp"

// Prediction results
struct InferenceResult {
    int prediction_class; // 0 = Hey Pluto, 1 = Unknown, 2 = Noise
    float probabilities[3];
};

// Initializes the inference engine and spawns InferenceTask
// feature_queue: Queue to pop filled FeatureTensor* from
// feature_free_queue: Queue to push processed FeatureTensor* back to
// decision_queue: Queue to push InferenceResult into
void inference_engine_init(QueueHandle_t feature_queue, QueueHandle_t feature_free_queue,
                           QueueHandle_t decision_queue);
