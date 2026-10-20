#include "rgb_led.hpp"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Define the single GPIO for the 2-pin RGB LED
#define RGB_TEST_PIN GPIO_NUM_2

void rgb_led_init(void) {
    gpio_reset_pin(RGB_TEST_PIN);
    gpio_set_direction(RGB_TEST_PIN, GPIO_MODE_OUTPUT);

    // Initial state: Off
    gpio_set_level(RGB_TEST_PIN, 0);
}

void rgb_led_set_mode(int mode) {
    if (mode == 2) {
        // Detected (LED ON)
        gpio_set_level(RGB_TEST_PIN, 1);
    } else {
        // Sleep (0) or Listening (1)
        // LED OFF
        gpio_set_level(RGB_TEST_PIN, 0);
    }
}

