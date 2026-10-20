#pragma once
#include <stdint.h>

// Call this once at startup (after boot, handles WiFi internally)
void ws_streamer_init(const char* wifi_ssid,
                      const char* wifi_pass,
                      const char* ws_uri);

// Call this from inference_engine.cpp when "Hey Pluto" is detected
// Pass the current timestamp in microseconds (esp_timer_get_time())
void ws_trigger_stream(int64_t timestamp_ms);
