#include "web_server.hpp"
#include <esp_http_server.h>
#include <esp_log.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>

static const char *TAG = "WEB_SERVER";

static esp_err_t download_wav_handler(httpd_req_t *req)
{
    FILE *f = fopen("/spiffs/mic_test.wav", "rb");
    if (f == NULL) {
        ESP_LOGE(TAG, "Failed to open mic_test.wav");
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "audio/wav");
    httpd_resp_set_hdr(req, "Content-Disposition", "attachment; filename=mic_test.wav");

    char chunk[1024];
    size_t chunksize = 0;
    while ((chunksize = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        if (httpd_resp_send_chunk(req, chunk, chunksize) != ESP_OK) {
            ESP_LOGE(TAG, "File sending failed!");
            fclose(f);
            return ESP_FAIL;
        }
    }
    
    // Send empty chunk to signal completion
    httpd_resp_send_chunk(req, NULL, 0);
    fclose(f);
    ESP_LOGI(TAG, "File sent successfully");
    
    return ESP_OK;
}

static const httpd_uri_t download_wav_uri = {
    .uri       = "/mic_test.wav",
    .method    = HTTP_GET,
    .handler   = download_wav_handler,
    .user_ctx  = NULL
};

void start_webserver(void)
{
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    // Start the httpd server
    ESP_LOGI(TAG, "Starting server on port: '%d'", config.server_port);
    if (httpd_start(&server, &config) == ESP_OK) {
        // Set URI handlers
        ESP_LOGI(TAG, "Registering URI handlers");
        httpd_register_uri_handler(server, &download_wav_uri);
    } else {
        ESP_LOGE(TAG, "Error starting server!");
    }
}
