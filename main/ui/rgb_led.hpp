#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the 2-pin RGB LED pin (GPIO 2)
void rgb_led_init(void);

// Set the 2-pin RGB LED state based on current system status
// Mode 0: Sleep (LED OFF)
// Mode 1: Listening (LED OFF)
// Mode 2: Detected (LED ON)
void rgb_led_set_mode(int mode);

#ifdef __cplusplus
}
#endif
