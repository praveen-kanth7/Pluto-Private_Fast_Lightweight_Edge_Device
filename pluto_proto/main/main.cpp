#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "audio/audio_capture.hpp"
#include "dsp/mfcc_processor.hpp"
#include "ml/inference_engine.hpp"
#include "app/wake_word_detector.hpp"
#include "diagnostics/pipeline_diagnostic.hpp"

#define DIAG_MODE 0 // Set to 1 to run pipeline diagnostic and halt

static const char *TAG = "Main";

static AudioFrame audio_pool[2];
static FeatureTensor feature_pool[2];

extern "C" void app_main() {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "       PLUTO WAKE WORD SYSTEM           ");
    ESP_LOGI(TAG, "========================================");

    if (DIAG_MODE == 1) {
        pipeline_diagnostic_run();
    }

    // Create queues
    QueueHandle_t pcm_free_queue = xQueueCreate(2, sizeof(AudioFrame*));
    QueueHandle_t pcm_queue = xQueueCreate(2, sizeof(AudioFrame*));
    
    QueueHandle_t feature_free_queue = xQueueCreate(2, sizeof(FeatureTensor*));
    QueueHandle_t feature_queue = xQueueCreate(2, sizeof(FeatureTensor*));
    
    QueueHandle_t decision_queue = xQueueCreate(5, sizeof(InferenceResult));

    // Seed the free queues with statically allocated buffers
    AudioFrame* p1 = &audio_pool[0];
    AudioFrame* p2 = &audio_pool[1];
    xQueueSend(pcm_free_queue, &p1, 0);
    xQueueSend(pcm_free_queue, &p2, 0);

    FeatureTensor* f1 = &feature_pool[0];
    FeatureTensor* f2 = &feature_pool[1];
    xQueueSend(feature_free_queue, &f1, 0);
    xQueueSend(feature_free_queue, &f2, 0);

    // Initialize modules and spawn their tasks
    audio_capture_init(pcm_queue, pcm_free_queue);
    dsp_processor_init(pcm_queue, pcm_free_queue, feature_queue, feature_free_queue);
    inference_engine_init(feature_queue, feature_free_queue, decision_queue);
    wake_word_detector_init(decision_queue);

    // Initial log of available memory
    ESP_LOGI(TAG, "Free heap: %u bytes", (unsigned)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free internal SRAM: %u bytes", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    // app_main can just idle or monitor
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "Free Heap: %u bytes, Internal SRAM: %u bytes", 
                 esp_get_free_heap_size(), 
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        // Optional: print high water marks, CPU load
    }
}
