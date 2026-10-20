#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the edge WebSocket client.
 * Connects to the EC2 server and sends device_info.
 */
void edge_client_init(void);

/**
 * @brief Trigger the streaming of audio due to a keyword detection.
 * 
 * @param keyword The keyword that was detected (e.g., "Hey Pluto").
 */
void edge_client_trigger_keyword(const char *keyword);

/**
 * @brief Push raw 16-bit PCM samples to the WebSocket audio stream.
 * 
 * @param data Pointer to raw PCM data.
 * @param size_bytes Size of the data in bytes.
 * @return true if data was pushed, false if not streaming or buffer full.
 */
bool edge_client_push_audio(const void *data, size_t size_bytes);

/**
 * @brief Check if the edge client is currently actively streaming audio.
 */
bool edge_client_is_streaming(void);

#ifdef __cplusplus
}
#endif
