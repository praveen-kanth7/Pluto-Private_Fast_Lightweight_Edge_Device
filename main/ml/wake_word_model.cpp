#include "wake_word_model.hpp"
#include "esp_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
// Tiny DS-CNN INT8 model — 9.8 KB, measured arena: 42.08 KB, production: 44 KB
#include "../tiny_ds_cnn_int8.h"

static const char *TAG = "Model";

const tflite::Model* wake_word_model_get() {
    const tflite::Model *model = tflite::GetModel(tiny_ds_cnn_int8_model);
    if (model == nullptr) {
        ESP_LOGE(TAG, "Failed to load model");
        abort();
    }
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch");
        abort();
    }
    return model;
}
