#include "oled_display.hpp"
#include "font5x7.h"
#include "driver/i2c.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>
#include <stdio.h>
#include "esp_heap_caps.h"

const uint8_t icon_moon[128] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf0, 0x38, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0xf8, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x80, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x3f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 0xfc, 0xf8, 0xf0,
  0xe0, 0xe0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xe0, 0xe0, 0x30, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x1f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
  0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x1f, 0x1f, 0x0f, 0x07, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00,
};

const uint8_t icon_mic[128] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0xf0, 0xf0, 0xf0,
  0xf0, 0xf0, 0xf0, 0xf0, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0xff, 0xff, 0xff, 0xff,
  0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7f, 0x40, 0x43, 0x47, 0x47, 0x47,
  0xc7, 0x47, 0x47, 0x47, 0x43, 0x40, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70,
  0x7f, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};



#define I2C_MASTER_SDA_IO           GPIO_NUM_8
#define I2C_MASTER_SCL_IO           GPIO_NUM_9
#define I2C_MASTER_NUM              I2C_NUM_0       
#define I2C_MASTER_FREQ_HZ          100000
#define OLED_I2C_ADDRESS            0x3C 

static const char *TAG = "OLED";

// Task handle — used by oled_set_state(DETECTED) to wake OLED immediately
static TaskHandle_t s_oled_handle = NULL;

static volatile oled_state_t current_state = OLED_STATE_SLEEP;
static volatile float g_dsp_util = 0.0f;
static volatile float g_ml_util = 0.0f;

static uint8_t display_buffer[8][128]; // 8 pages, 128 segments (columns)

static void oled_draw_icon(int start_x, int start_y_page, const uint8_t* icon) {
    if (start_x < 0 || start_x + 32 > 128 || start_y_page < 0 || start_y_page + 4 > 8) return;
    for (int p = 0; p < 4; p++) {
        for (int x = 0; x < 32; x++) {
            display_buffer[start_y_page + p][start_x + x] = icon[p * 32 + x];
        }
    }
}

static void send_command(uint8_t cmd_byte) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (OLED_I2C_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, 0x00, true);
    i2c_master_write_byte(cmd, cmd_byte, true);
    i2c_master_stop(cmd);
    i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);
}

static void oled_update_screen() {
    // Re-send critical initialization commands to recover from potential voltage dips or I2C glitches
    send_command(0xAF); // Display ON
    send_command(0xAD); // SH1106 DC-DC Control Mode
    send_command(0x8B); // SH1106 DC-DC ON
    send_command(0x8D); // SSD1306 Charge Pump Setting
    send_command(0x14); // SSD1306 Enable Charge Pump
    
    // Also re-send orientation commands in case of a reset, 
    // otherwise it will default to being upside-down!
    send_command(0xA1); // Segment remap
    send_command(0xC8); // COM scan direction

    for (int page = 0; page < 8; page++) {
        send_command(0xB0 + page);
        send_command(0x02); // Offset by 2 for 1.3" screen (SH1106)
        send_command(0x10);

        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (OLED_I2C_ADDRESS << 1) | I2C_MASTER_WRITE, true);
        i2c_master_write_byte(cmd, 0x40, true);
        i2c_master_write(cmd, display_buffer[page], 128, true);
        i2c_master_stop(cmd);
        i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(1000));
        i2c_cmd_link_delete(cmd);
    }
}

static void oled_clear_buffer() {
    memset(display_buffer, 0, sizeof(display_buffer));
}

static void oled_draw_char(int x, int y_page, char c) {
    if (x < 0 || x > 122 || y_page < 0 || y_page > 7) return;
    for (int i = 0; i < 5; i++) {
        display_buffer[y_page][x + i] = font5x7[(uint8_t)c][i];
    }
}

static void oled_draw_string(int x, int y_page, const char* str) {
    int cx = x;
    while (*str) {
        oled_draw_char(cx, y_page, *str);
        cx += 6; // 5 width + 1 space
        if (cx > 122) break;
        str++;
    }
}

static void oled_task(void *arg) {
    // Register handle FIRST so oled_set_state can notify us immediately
    s_oled_handle = xTaskGetCurrentTaskHandle();
    ESP_LOGI(TAG, "OLED Task started on Core %d", xPortGetCoreID());
    
    // Initialize I2C
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master = {
            .clk_speed = I2C_MASTER_FREQ_HZ,
        },
        .clk_flags = 0,
    };
    i2c_param_config(I2C_MASTER_NUM, &conf);
    i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);

    // Initialize SH1106
    uint8_t init_cmds[] = {
        0xAE, 0x02, 0x10, 0x40, 0xB0, 0x81, 0x80, 0xA1, 0xA6, 0xA8, 0x3F, 
        0xAD, 0x8B, 0x33, 0xC8, 0xD3, 0x00, 0xD5, 0x80, 0xD9, 0x1F, 
        0xDA, 0x12, 0xDB, 0x40, 0xAF
    };
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (OLED_I2C_ADDRESS << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, 0x00, true);
    i2c_master_write(cmd, init_cmds, sizeof(init_cmds), true);
    i2c_master_stop(cmd);
    i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);

    oled_clear_buffer();
    oled_update_screen();

    int detected_hold_frames = 0;
    int silence_frames = 0;

    while (1) {
        // Block up to 100ms OR wake immediately when DETECTED notification arrives.
        // xTaskNotify (from oled_set_state) uses FreeRTOS IPC with proper memory
        // barriers — guaranteed cross-core visibility, no cache coherency issues.
        uint32_t notif_val = 0;
        xTaskNotifyWait(0, ULONG_MAX, &notif_val, pdMS_TO_TICKS(100));

        // DETECTED signal received via task notification — trigger immediately
        if (notif_val != 0) {
            detected_hold_frames = 30; // 3 seconds hold
            silence_frames = 0;
            current_state = OLED_STATE_LISTENING;
            ESP_LOGI(TAG, "[OLED] DETECTED notification received — showing banner");
        }

        oled_clear_buffer();
        if (detected_hold_frames > 0) {
            oled_draw_string(20, 4, "streaming....");
            detected_hold_frames--;
        } else {
            // Normal state display
            oled_state_t state = current_state;

            if (state == OLED_STATE_SLEEP) {
                silence_frames++;
            } else {
                silence_frames = 0;
            }

            oled_state_t display_state = (silence_frames >= 20) ? OLED_STATE_SLEEP : OLED_STATE_LISTENING;

            // RAM consumption at the top left corner
            uint32_t total_heap = heap_caps_get_total_size(MALLOC_CAP_8BIT);
            uint32_t free_heap = heap_caps_get_free_size(MALLOC_CAP_8BIT);
            uint32_t dynamic_used = total_heap - free_heap;
            uint32_t kws_used_ram = dynamic_used + (78 * 1024);
            char mem_buf[16];
            snprintf(mem_buf, sizeof(mem_buf), "R:%luKB", (unsigned long)(kws_used_ram / 1024));
            oled_draw_string(0, 0, mem_buf);

            // C0 utilization in the left down corner
            char c0_buf[16];
            snprintf(c0_buf, sizeof(c0_buf), "C0:%.1f%%", g_dsp_util);
            oled_draw_string(0, 7, c0_buf);

            // C1 utilization in the right down corner
            char c1_buf[16];
            snprintf(c1_buf, sizeof(c1_buf), "C1:%.1f%%", g_ml_util);
            int c1_len = strlen(c1_buf);
            oled_draw_string(128 - (c1_len * 6), 7, c1_buf);

            if (display_state == OLED_STATE_SLEEP) {
                oled_draw_icon(48, 2, icon_moon);
                
                static int zz_count = 0;
                static int zz_frame = 0;
                zz_frame++;
                if (zz_frame % 8 == 0) {
                    zz_count = (zz_count + 1) % 4;
                }
                
                if (zz_count >= 1) oled_draw_string(80, 4, "z");
                if (zz_count >= 2) oled_draw_string(88, 3, "Z");
                if (zz_count >= 3) oled_draw_string(96, 2, "z");
            } else {
                oled_draw_icon(48, 1, icon_mic);
                
                static int dot_count = 0;
                static int frame_tick = 0;
                frame_tick++;
                if (frame_tick % 5 == 0) {
                    dot_count = (dot_count + 1) % 4;
                }
                char listen_buf[32];
                strcpy(listen_buf, "listening mode");
                for (int i=0; i<dot_count; i++) strcat(listen_buf, ".");
                
                int listen_len = strlen(listen_buf);
                oled_draw_string(64 - (listen_len * 3), 5, listen_buf);
            }
        }

        // CPU activity bar at the top row
        float total_util = g_dsp_util + g_ml_util;
        int bars = (int)(total_util * 128.0f / 200.0f);
        if (bars > 128) bars = 128;
        for (int i = 0; i < bars; i++) {
            display_buffer[0][i] = 0x3C;
        }

        oled_update_screen();
    }
}

void oled_init(void) {
    xTaskCreatePinnedToCore(
        oled_task,
        "OLED",
        4096,
        NULL,
        configMAX_PRIORITIES - 4,
        NULL,
        0 // Core 0: keeps OLED away from Inference task (Core 1)
          // Inference is prio MAX-3 on Core 1; DSP is prio MAX-2 on Core 0
          // OLED runs at MAX-4 and gets CPU between DSP notification waits
    );
}

void oled_set_state(oled_state_t new_state) {
    current_state = new_state;
    if (new_state == OLED_STATE_DETECTED && s_oled_handle != NULL) {
        // Wake the OLED task IMMEDIATELY via FreeRTOS notification.
        // This crosses core boundaries with proper memory barriers —
        // no cache coherency issues, no 100ms polling blind spot.
        xTaskNotify(s_oled_handle, 1, eSetValueWithOverwrite);
    }
}

void oled_set_cpu_utilization(float dsp_util, float ml_util) {
    g_dsp_util = dsp_util;
    g_ml_util = ml_util;
}
