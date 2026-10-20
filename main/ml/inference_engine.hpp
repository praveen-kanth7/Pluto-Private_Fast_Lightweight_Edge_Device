#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/FreeRTOS.h"

// Prediction results
struct InferenceResult {
    int prediction_class; // 0 = Hey Pluto, 1 = Unknown
    float probabilities[2]; 

};

// Initializes the TFLite inference engine and creates the InferenceTask.
void inference_engine_init();
