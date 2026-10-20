#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_spiffs.h"

#include "dsp/continuous_dsp.hpp"
#include "ml/inference_engine.hpp"
#include "ui/oled_display.hpp"
#include "ui/rgb_led.hpp"
#include "net/wifi_manager.hpp"
#include "net/web_server.hpp"
#include "ws_streamer.hpp"
#include "esp_netif.h"

static const char *TAG = "MAIN";

extern "C" void app_main() {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "    PLUTO WAKE WORD SYSTEM (CONTINUOUS) ");
    ESP_LOGI(TAG, "========================================");

    // Initialize the RGB LED
    rgb_led_init();

    // Initialize the UI Task
    oled_init();

    // Initialize SPIFFS for WAV recording
    esp_vfs_spiffs_conf_t spiffs_conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };
    esp_err_t ret = esp_vfs_spiffs_register(&spiffs_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "SPIFFS mounted successfully at /spiffs");
    }

    // Initialize the two continuous sliding-window tasks FIRST
    // (DSP must start before ws_streamer so rx_handle_global is set)
    inference_engine_init();
    continuous_dsp_init();

    // Initialize WiFi and Web Server to serve mic_test.wav
    wifi_init_sta();
    start_webserver();


    // Initial log of available memory
    ESP_LOGI(TAG, "Free heap: %u bytes", (unsigned)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free internal SRAM: %u bytes", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));

    // Monitor loop
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGI(TAG, "Free Heap: %u bytes, Internal SRAM: %u bytes", 
                 esp_get_free_heap_size(), 
                 heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
                 
        esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
        if (netif) {
            esp_netif_ip_info_t ip_info;
            esp_netif_get_ip_info(netif, &ip_info);
            if (ip_info.ip.addr != 0) {
                ESP_LOGI(TAG, "Web Server is running! Download audio at: http://" IPSTR "/mic_test.wav", IP2STR(&ip_info.ip));
            } else {
                ESP_LOGW(TAG, "WiFi not connected. Check SSID/Password in wifi_manager.cpp!");
            }
        }
    }
}
