#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize and connect to the WiFi network in Station (STA) mode.
 * 
 * NOTE: Ensure you have initialized NVS before calling this if it hasn't 
 * been initialized elsewhere.
 */
void wifi_init_sta(void);

#ifdef __cplusplus
}
#endif
