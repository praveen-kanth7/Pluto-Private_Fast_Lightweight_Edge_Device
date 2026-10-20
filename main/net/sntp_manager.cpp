#include "sntp_manager.hpp"
#include "esp_sntp.h"
#include "esp_netif_sntp.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "SNTP";

void sntp_init_sync(void) {
    ESP_LOGI(TAG, "Initializing SNTP");
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "pool.ntp.org");
    esp_sntp_init();

    // Wait for time to be set
    int retry = 0;
    const int retry_count = 15;
    while (esp_sntp_get_sync_status() == SNTP_SYNC_STATUS_RESET && ++retry < retry_count) {
        ESP_LOGI(TAG, "Waiting for system time to be set... (%d/%d)", retry, retry_count);
        vTaskDelay(2000 / portTICK_PERIOD_MS);
    }
    
    if (retry == retry_count) {
        ESP_LOGE(TAG, "Failed to synchronize time via SNTP!");
    } else {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        ESP_LOGI(TAG, "Time synchronized! Epoch: %lld", (long long)tv.tv_sec);
    }
}
