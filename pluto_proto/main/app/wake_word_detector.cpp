#include "wake_word_detector.hpp"
#include "../ml/inference_engine.hpp"
#include "esp_log.h"

static const char *TAG = "App";

static QueueHandle_t in_decision;

static void decision_task(void *arg) {
    ESP_LOGI(TAG, "DecisionTask started on Core %d", xPortGetCoreID());

    while (1) {
        InferenceResult result;
        if (xQueueReceive(in_decision, &result, portMAX_DELAY) == pdTRUE) {
            
            ESP_LOGI(TAG, "Prediction: class=%d, prob=[%.6f, %.6f, %.6f] | HighWaterMark: %d words", 
                     result.prediction_class, result.probabilities[0], 
                     result.probabilities[1], result.probabilities[2],
                     uxTaskGetStackHighWaterMark(NULL));

            if (result.prediction_class == 0) {
                ESP_LOGI(TAG, ">>> HEY PLUTO DETECTED <<<");
            } else if (result.prediction_class == 1) {
                ESP_LOGI(TAG, "UNKNOWN WORD");
            } else {
                ESP_LOGI(TAG, "NOISE");
            }
        }
    }
}

void wake_word_detector_init(QueueHandle_t decision_queue) {
    in_decision = decision_queue;

    xTaskCreatePinnedToCore(
        decision_task,
        "Decision",
        4096,
        NULL,
        configMAX_PRIORITIES - 4, // Lowest priority in pipeline
        NULL,
        0 // Core 0
    );
}
