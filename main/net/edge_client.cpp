#include "edge_client.hpp"
#include "esp_websocket_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/stream_buffer.h"
#include "cJSON.h"
#include <sys/time.h>
#include <string.h>

static const char *TAG = "EdgeClient";

#define WS_URI "ws://34.197.0.214:8000/ws/edge"
#define DEVICE_ID "ESP32-S3-001"
#define AUDIO_CHUNK_SIZE 1024
#define STREAM_DURATION_US 5000000LL // 5 seconds

static esp_websocket_client_handle_t ws_client = NULL;
static StreamBufferHandle_t audio_stream_buf = NULL;
static volatile bool is_streaming = false;
static int64_t stream_stop_time_us = 0;
static bool ws_connected = false;

static void websocket_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *data = (esp_websocket_event_data_t *)event_data;
    
    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "WEBSOCKET_EVENT_CONNECTED");
            ws_connected = true;
            
            // STEP 1: Send device_info immediately upon connection
            {
                cJSON *root = cJSON_CreateObject();
                cJSON_AddStringToObject(root, "type", "device_info");
                cJSON_AddStringToObject(root, "device_id", DEVICE_ID);
                cJSON_AddStringToObject(root, "firmware", "prototype-1.0");
                cJSON_AddNumberToObject(root, "sample_rate", 16000);
                cJSON_AddNumberToObject(root, "bits", 16);
                cJSON_AddNumberToObject(root, "channels", 1);
                
                char *json_str = cJSON_PrintUnformatted(root);
                esp_websocket_client_send_text(ws_client, json_str, strlen(json_str), portMAX_DELAY);
                ESP_LOGI(TAG, "Sent device_info: %s", json_str);
                
                cJSON_Delete(root);
                free(json_str);
            }
            break;
            
        case WEBSOCKET_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "WEBSOCKET_EVENT_DISCONNECTED");
            ws_connected = false;
            is_streaming = false;
            break;
            
        default:
            break;
    }
}

static void edge_client_task(void *arg) {
    uint8_t *chunk_buffer = (uint8_t *)malloc(AUDIO_CHUNK_SIZE);
    
    while (1) {
        if (!ws_connected) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        if (is_streaming) {
            int64_t now = esp_timer_get_time();
            
            // STEP 5: Stop streaming if duration exceeded
            if (now >= stream_stop_time_us) {
                is_streaming = false;
                
                cJSON *root = cJSON_CreateObject();
                cJSON_AddStringToObject(root, "type", "stream_end");
                cJSON_AddStringToObject(root, "device_id", DEVICE_ID);
                
                char *json_str = cJSON_PrintUnformatted(root);
                esp_websocket_client_send_text(ws_client, json_str, strlen(json_str), portMAX_DELAY);
                ESP_LOGI(TAG, "Sent stream_end");
                
                cJSON_Delete(root);
                free(json_str);
                
                // Clear any residual data in the buffer
                xStreamBufferReset(audio_stream_buf);
                continue;
            }
            
            // STEP 4: Send audio chunks
            size_t bytes_received = xStreamBufferReceive(audio_stream_buf, chunk_buffer, AUDIO_CHUNK_SIZE, pdMS_TO_TICKS(50));
            if (bytes_received == AUDIO_CHUNK_SIZE) {
                esp_websocket_client_send_bin(ws_client, (const char *)chunk_buffer, AUDIO_CHUNK_SIZE, portMAX_DELAY);
            }
        } else {
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
    
    free(chunk_buffer);
    vTaskDelete(NULL);
}

void edge_client_init(void) {
    audio_stream_buf = xStreamBufferCreate(AUDIO_CHUNK_SIZE * 4, AUDIO_CHUNK_SIZE);
    if (audio_stream_buf == NULL) {
        ESP_LOGE(TAG, "Failed to create StreamBuffer");
        return;
    }

    esp_websocket_client_config_t ws_cfg = {};
    ws_cfg.uri = WS_URI;

    ws_client = esp_websocket_client_init(&ws_cfg);
    esp_websocket_register_events(ws_client, WEBSOCKET_EVENT_ANY, websocket_event_handler, (void *)ws_client);
    esp_websocket_client_start(ws_client);

    xTaskCreate(edge_client_task, "edge_client_task", 4096, NULL, 5, NULL);
}

void edge_client_trigger_keyword(const char *keyword) {
    if (!ws_connected || is_streaming) return;

    // Get SNTP epoch time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64_t timestamp_ms = (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;

    // STEP 2: Keyword detected
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "type", "keyword_detected");
    cJSON_AddStringToObject(root, "device_id", DEVICE_ID);
    cJSON_AddStringToObject(root, "keyword", keyword);
    cJSON_AddNumberToObject(root, "timestamp_ms", timestamp_ms);
    
    char *json_str = cJSON_PrintUnformatted(root);
    esp_websocket_client_send_text(ws_client, json_str, strlen(json_str), portMAX_DELAY);
    ESP_LOGI(TAG, "Sent keyword_detected: %s", json_str);
    cJSON_Delete(root);
    free(json_str);

    // STEP 3: Stream start
    root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "type", "stream_start");
    cJSON_AddStringToObject(root, "device_id", DEVICE_ID);
    cJSON_AddNumberToObject(root, "sample_rate", 16000);
    cJSON_AddNumberToObject(root, "bits", 16);
    cJSON_AddNumberToObject(root, "channels", 1);

    json_str = cJSON_PrintUnformatted(root);
    esp_websocket_client_send_text(ws_client, json_str, strlen(json_str), portMAX_DELAY);
    ESP_LOGI(TAG, "Sent stream_start");
    cJSON_Delete(root);
    free(json_str);

    xStreamBufferReset(audio_stream_buf);
    stream_stop_time_us = esp_timer_get_time() + STREAM_DURATION_US;
    is_streaming = true;
}

bool edge_client_push_audio(const void *data, size_t size_bytes) {
    if (!is_streaming || audio_stream_buf == NULL) return false;
    
    // We don't block here to avoid slowing down the DSP task.
    // If the buffer is full, we drop samples (network is too slow).
    size_t bytes_sent = xStreamBufferSend(audio_stream_buf, data, size_bytes, 0);
    return (bytes_sent == size_bytes);
}

bool edge_client_is_streaming(void) {
    return is_streaming;
}
