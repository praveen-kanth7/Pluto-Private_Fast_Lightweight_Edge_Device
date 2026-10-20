#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// The external handle to the Inference task so we can notify it
extern TaskHandle_t inference_task_handle;

// The global ring buffer for Mel energies (49 frames, 40 bins)
// It's exposed so the inference task can read the last 49 frames.
#define MFCC_FRAMES 49
#define MEL_FILTERS 40
extern float mel_ring_buffer[MFCC_FRAMES][MEL_FILTERS];

// The current head index of the ring buffer
extern volatile int mel_ring_head;

void continuous_dsp_init();
