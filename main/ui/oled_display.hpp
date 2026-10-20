#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    OLED_STATE_SLEEP = 0,
    OLED_STATE_LISTENING,
    OLED_STATE_DETECTED
} oled_state_t;

void oled_init(void);
void oled_set_state(oled_state_t new_state);
void oled_set_cpu_utilization(float dsp_util, float ml_util);

#ifdef __cplusplus
}
#endif
