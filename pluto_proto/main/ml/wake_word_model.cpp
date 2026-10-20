#include "wake_word_model.hpp"
#include "esp_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

static const char *TAG = "Model";

extern const uint8_t ds_cnn_int8_tflite_start[] asm("_binary_ds_cnn_int8_tflite_start");
extern const uint8_t ds_cnn_int8_tflite_end[] asm("_binary_ds_cnn_int8_tflite_end");

const tflite::Model* wake_word_model_get() {
    const tflite::Model *model = tflite::GetModel(ds_cnn_int8_tflite_start);
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
