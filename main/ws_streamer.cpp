#include "ws_streamer.hpp"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_websocket_client.h"
#include "driver/i2s_std.h"
#include <string.h>
#include <stdio.h>

static const char* TAG = "WS_STREAMER";

// I2S handle — exposed from continuous_dsp.cpp
extern i2s_chan_handle_t rx_handle_global;

static const char* g_ws_uri = NULL;
static esp_websocket_client_handle_t g_ws_client = NULL;
static QueueHandle_t g_trigger_queue = NULL;

typedef struct {
    int64_t timestamp_ms;
} trigger_msg_t;

// ── WiFi ─────────────────────────────────────────────────────
static void wifi_event_handler(void* arg, esp_event_base_t base,
                                int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disconnected, reconnecting...");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*)data;
        ESP_LOGI(TAG, "WiFi connected! IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

static void wifi_init(const char* ssid, const char* pass) {
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT,   IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);

    wifi_config_t wifi_cfg = {};
    strncpy((char*)wifi_cfg.sta.ssid,     ssid, sizeof(wifi_cfg.sta.ssid));
    strncpy((char*)wifi_cfg.sta.password, pass, sizeof(wifi_cfg.sta.password));
    wifi_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg);
    esp_wifi_start();
    esp_wifi_connect();

    ESP_LOGI(TAG, "WiFi connecting to: %s", ssid);
    vTaskDelay(pdMS_TO_TICKS(5000)); // Wait for connection
}

// ── Streaming Task ────────────────────────────────────────────
static void stream_task(void* arg) {
    trigger_msg_t msg;

    while (1) {
        if (xQueueReceive(g_trigger_queue, &msg, portMAX_DELAY) != pdTRUE)
            continue;

        ESP_LOGI(TAG, "Trigger received — connecting WebSocket...");

        esp_websocket_client_config_t ws_cfg = {};
        ws_cfg.uri = g_ws_uri;
        ws_cfg.port = 80;
        ws_cfg.reconnect_timeout_ms = 2000;
        ws_cfg.network_timeout_ms = 10000;

        g_ws_client = esp_websocket_client_init(&ws_cfg);
        esp_websocket_client_start(g_ws_client);

        // Wait up to 5 seconds (50 × 100ms) for WebSocket connection
        int wait_retries = 0;
        while (!esp_websocket_client_is_connected(g_ws_client) && wait_retries < 50) {
            vTaskDelay(pdMS_TO_TICKS(100));
            wait_retries++;
        }

        if (!esp_websocket_client_is_connected(g_ws_client)) {
            ESP_LOGE(TAG, "WebSocket connection TIMED OUT");
            esp_websocket_client_destroy(g_ws_client);
            g_ws_client = NULL;
            continue;
        }
        ESP_LOGI(TAG, "WebSocket connected!");

        // ── STEP 1: device_info ──────────────────────────────
        char device_info[256];
        snprintf(device_info, sizeof(device_info),
            "{\"type\":\"device_info\","
            "\"device_id\":\"ESP32-S3-001\","
            "\"firmware\":\"prototype-1.0\","
            "\"sample_rate\":16000,"
            "\"bits\":16,"
            "\"channels\":1}");
        esp_websocket_client_send_text(g_ws_client, device_info,
                                       strlen(device_info), portMAX_DELAY);
        ESP_LOGI(TAG, "Sent: device_info");

        // ── STEP 2: keyword_detected ─────────────────────────
        char kw_msg[256];
        snprintf(kw_msg, sizeof(kw_msg),
            "{\"type\":\"keyword_detected\","
            "\"device_id\":\"ESP32-S3-001\","
            "\"keyword\":\"Hey Pluto\","
            "\"timestamp_ms\":%lld}", msg.timestamp_ms);
        esp_websocket_client_send_text(g_ws_client, kw_msg,
                                       strlen(kw_msg), portMAX_DELAY);
        ESP_LOGI(TAG, "Sent: keyword_detected");

        // ── STEP 3: stream_start ─────────────────────────────
        const char* stream_start =
            "{\"type\":\"stream_start\","
            "\"device_id\":\"ESP32-S3-001\","
            "\"sample_rate\":16000,"
            "\"bits\":16,"
            "\"channels\":1}";
        esp_websocket_client_send_text(g_ws_client, stream_start,
                                       strlen(stream_start), portMAX_DELAY);
        ESP_LOGI(TAG, "Sent: stream_start");

        // ── STEP 4: Stream raw PCM audio (3 seconds) ─────────
        // 16kHz x 2 bytes x 3 seconds = 96,000 bytes total
        // 512 samples x sizeof(int16) = 1024 bytes per chunk
        #define CHUNK_SAMPLES  512
        #define STREAM_SECONDS 3
        #define TOTAL_CHUNKS   (16000 * STREAM_SECONDS / CHUNK_SAMPLES)

        // Read 1024 stereo samples to extract 512 mono samples
        static int32_t stereo32[CHUNK_SAMPLES * 2];
        static int16_t pcm16[CHUNK_SAMPLES];
        size_t bytes_read = 0;

        for (int chunk = 0; chunk < TOTAL_CHUNKS; chunk++) {
            i2s_channel_read(rx_handle_global,
                             stereo32,
                             (CHUNK_SAMPLES * 2) * sizeof(int32_t),
                             &bytes_read,
                             portMAX_DELAY);

            int stereo_samples = bytes_read / sizeof(int32_t);
            int mono_samples = stereo_samples / 2;
            
            for (int i = 0; i < mono_samples; i++) {
                // Extract only the Left channel (Even indices) and convert to 16-bit
                pcm16[i] = (int16_t)(stereo32[i * 2] >> 16);
            }

            esp_websocket_client_send_bin(g_ws_client,
                                          (const char*)pcm16,
                                          mono_samples * sizeof(int16_t),
                                          portMAX_DELAY);

            if (chunk % 10 == 0) {
                ESP_LOGI(TAG, "Streaming chunk %d/%d", chunk, TOTAL_CHUNKS);
            }

            vTaskDelay(pdMS_TO_TICKS(32)); // ~32ms per chunk = real-time
        }

        // ── STEP 5: stream_end ───────────────────────────────
        const char* stream_end =
            "{\"type\":\"stream_end\","
            "\"device_id\":\"ESP32-S3-001\"}";
        esp_websocket_client_send_text(g_ws_client, stream_end,
                                       strlen(stream_end), portMAX_DELAY);
        ESP_LOGI(TAG, "Sent: stream_end");

        vTaskDelay(pdMS_TO_TICKS(500));
        esp_websocket_client_stop(g_ws_client);
        esp_websocket_client_destroy(g_ws_client);
        g_ws_client = NULL;
        ESP_LOGI(TAG, "Streaming complete!");
    }
}

// ── Public API ────────────────────────────────────────────────
void ws_streamer_init(const char* wifi_ssid,
                      const char* wifi_pass,
                      const char* ws_uri) {
    g_ws_uri = ws_uri;
    wifi_init(wifi_ssid, wifi_pass);

    g_trigger_queue = xQueueCreate(1, sizeof(trigger_msg_t));

    xTaskCreatePinnedToCore(
        stream_task,
        "WS_Stream",
        8192,
        NULL,
        5,
        NULL,
        1  // Core 1
    );

    ESP_LOGI(TAG, "WS Streamer initialized");
}

void ws_trigger_stream(int64_t timestamp_ms) {
    if (g_trigger_queue == NULL) return;
    trigger_msg_t msg = { .timestamp_ms = timestamp_ms };
    xQueueOverwrite(g_trigger_queue, &msg);
}
