#include "inference_engine.hpp"
#include "wake_word_model.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <string.h>

static const char *TAG = "Inference";

#define TENSOR_ARENA_SIZE (128 * 1024)
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

static QueueHandle_t in_feature;
static QueueHandle_t free_feature;
static QueueHandle_t out_decision;

static void inference_task(void *arg) {
    ESP_LOGI(TAG, "InferenceTask started on Core %d", xPortGetCoreID());

    while (1) {
        FeatureTensor *feature = nullptr;
        if (xQueueReceive(in_feature, &feature, portMAX_DELAY) == pdTRUE) {
            
            // Copy features to model input tensor
            memcpy(input_tensor->data.int8, feature->data, MFCC_TENSOR_SIZE);
            
            int64_t start = esp_timer_get_time();
            TfLiteStatus status = interpreter->Invoke();
            int64_t end = esp_timer_get_time();

            if (status != kTfLiteOk) {
                ESP_LOGE(TAG, "TFLite Invoke failed");
            } else {
                ESP_LOGI(TAG, "Inference executed in %lld us | HighWaterMark: %d words", 
                         end - start, uxTaskGetStackHighWaterMark(NULL));
                
                InferenceResult result;
                int8_t raw0 = output_tensor->data.int8[0];
                int8_t raw1 = output_tensor->data.int8[1];
                int8_t raw2 = output_tensor->data.int8[2];

                result.probabilities[0] = ((int)raw0 - output_tensor->params.zero_point) * output_tensor->params.scale;
                result.probabilities[1] = ((int)raw1 - output_tensor->params.zero_point) * output_tensor->params.scale;
                result.probabilities[2] = ((int)raw2 - output_tensor->params.zero_point) * output_tensor->params.scale;

                result.prediction_class = 0;
                float best = result.probabilities[0];
                if (result.probabilities[1] > best) { best = result.probabilities[1]; result.prediction_class = 1; }
                if (result.probabilities[2] > best) { best = result.probabilities[2]; result.prediction_class = 2; }
                
                // Send result to DecisionTask (by value, since it's small)
                xQueueSend(out_decision, &result, 0);
            }
            
            // Return feature tensor to free queue
            xQueueSend(free_feature, &feature, 0);
        }
    }
}

void inference_engine_init(QueueHandle_t feature_queue, QueueHandle_t feature_free_queue,
                           QueueHandle_t decision_queue) {
    in_feature = feature_queue;
    free_feature = feature_free_queue;
    out_decision = decision_queue;

    const tflite::Model *model = wake_word_model_get();

    static tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed");
        abort();
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    ESP_LOGI(TAG, "TFLite initialization complete, Arena Used: %d bytes", interpreter->arena_used_bytes());

    xTaskCreatePinnedToCore(
        inference_task,
        "Inference",
        8192,
        NULL,
        configMAX_PRIORITIES - 3, // Low priority compared to audio/dsp
        NULL,
        1 // Core 1
    );
}
