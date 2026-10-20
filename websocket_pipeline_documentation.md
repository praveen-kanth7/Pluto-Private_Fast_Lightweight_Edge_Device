# Pluto Wake Word System: WebSocket Streaming Pipeline

This document outlines exactly what happens *after* the Machine Learning model successfully detects the "Hey Pluto" wake word. This process handles notifying the cloud server and streaming the user's spoken command in real-time.

---

### 1. The Trigger
* **The Hand-off:** When `inference_engine.cpp` detects the wake word with >92% confidence, it immediately calls `ws_trigger_stream(timestamp_ms)`.
* **FreeRTOS Queue:** This function pushes the timestamp into a FreeRTOS Queue (`g_trigger_queue`). This queue safely passes the message to the background WebSocket task running independently on Core 1.

### 2. Establishing the Connection
* **Wake Up:** The `stream_task` (which is normally asleep and using 0% CPU) wakes up the instant the queue receives the message.
* **WebSocket Handshake:** The ESP32 immediately spins up a WebSocket client and negotiates a connection to your cloud server at `ws://34.197.0.214/ws/edge`. It will wait up to 5 seconds for the connection to succeed.

### 3. Sending Metadata (JSON)
Once connected, the ESP32 sends three JSON text packets in rapid succession to prep the server:
1. **`device_info`**: Tells the server the Device ID (`ESP32-S3-001`), firmware version, and the audio specifications it should expect (16000 Hz, 16-bit Mono).
2. **`keyword_detected`**: Officially informs the server that a wake word was spoken, passing along the exact millisecond timestamp of the event.
3. **`stream_start`**: Acts as a flag to tell the server that all future packets will be raw binary audio data instead of JSON text.

### 4. Real-Time Audio Streaming (Binary)
For the next exactly **3 seconds**, the ESP32 streams the user's spoken command to the cloud:
* **Microphone Hijacking:** The WebSocket task borrows the I2S microphone handle (`rx_handle_global`) to read audio directly from the hardware.
* **Chunking:** It reads audio in tiny chunks of `512 samples` (about 32 milliseconds of audio).
* **Down-sampling:** It converts the raw 32-bit microphone data into standard 16-bit PCM.
* **Binary Transmission:** It fires the 1024-byte chunk up to the cloud server over the WebSocket using binary frames (`esp_websocket_client_send_bin`).
* **Real-time Loop:** It loops this process roughly 93 times, sleeping for exactly 32ms between each chunk so that the audio streams to the server in perfect real-time.

### 5. Shutting Down
* **`stream_end`**: After 3 seconds of audio has been streamed, the ESP32 sends one final JSON packet (`stream_end`) to tell the server the recording is over.
* **Cleanup:** It safely closes the WebSocket connection, destroys the client to free up RAM, and goes back to sleep waiting for the next "Hey Pluto" trigger.
