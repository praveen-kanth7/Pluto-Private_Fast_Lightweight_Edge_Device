idf_component_register(
    SRCS "main.cpp"
         "dsp/fft480.cpp"
         "dsp/mel_filterbank.cpp"
            "dsp/continuous_dsp.cpp"
         "ml/wake_word_model.cpp"
         "ml/inference_engine.cpp"
         "diagnostics/pipeline_diagnostic.cpp"
         "net/wifi_manager.cpp"
         "net/web_server.cpp"
         "ui/oled_display.cpp"
         "ui/rgb_led.cpp"
         "ws_streamer.cpp"
    INCLUDE_DIRS "."
        REQUIRES esp_websocket_client json esp_netif lwip esp_driver_gpio esp_driver_i2c esp_driver_i2s esp_wifi nvs_flash spiffs esp_http_server
    PRIV_REQUIRES driver

)
