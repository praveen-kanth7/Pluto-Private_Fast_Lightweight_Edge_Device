#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// Initializes the DecisionTask
// decision_queue: Queue to pop InferenceResult from
void wake_word_detector_init(QueueHandle_t decision_queue);
