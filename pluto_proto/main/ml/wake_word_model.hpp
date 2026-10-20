#pragma once

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Returns the pre-loaded ds_cnn_int8 model
const tflite::Model* wake_word_model_get();
