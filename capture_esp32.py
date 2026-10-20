import serial
import time
import struct

PORT = "COM13"
BAUD = 115200

NUM_SAMPLES = 16000

START_MARKER = b"PCM_HEX_START\n"
END_MARKER = b"\nPCM_HEX_END"


print("Opening", PORT)

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

# Allow ESP32 to reset and start
time.sleep(2)

# Clear anything already in the serial buffer
# ser.reset_input_buffer()

print("Waiting for PCM_HEX_START...")

buffer = bytearray()
start_time = time.time()

while True:

    data = ser.read(4096)

    if data:
        buffer.extend(data)

        pos = buffer.find(START_MARKER)

        if pos != -1:
            print("Found PCM_HEX_START")

            buffer = buffer[
                pos + len(START_MARKER):
            ]

            break

    if time.time() - start_time > 20:
        print("ERROR: Timeout waiting for PCM_HEX_START")
        ser.close()
        raise SystemExit(1)


print("Receiving PCM samples...")

while END_MARKER not in buffer:

    data = ser.read(4096)

    if data:
        buffer.extend(data)

    if time.time() - start_time > 60:
        print("ERROR: Timeout waiting for PCM_HEX_END")
        ser.close()
        raise SystemExit(1)


ser.close()


# ------------------------------------------------------------
# Extract only PCM hex section
# ------------------------------------------------------------

end_pos = buffer.find(END_MARKER)

hex_data = bytes(
    buffer[:end_pos]
).strip()


# ------------------------------------------------------------
# Convert hex text to int16
# ------------------------------------------------------------

hex_values = hex_data.split(b",")

if len(hex_values) != NUM_SAMPLES:

    print(
        "ERROR: Expected",
        NUM_SAMPLES,
        "samples but received",
        len(hex_values)
    )

    raise SystemExit(1)


samples = []

for h in hex_values:

    value = int(h, 16)

    # Convert unsigned 16-bit to signed int16
    if value >= 32768:
        value -= 65536

    samples.append(value)


# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print()
print("================================")
print("ESP32 PCM CAPTURE SUCCESSFUL")
print("================================")

print("Samples:", len(samples))
print("Bytes:", len(samples) * 2)

print()
print("First 20 samples:")
print(samples[:20])

print()
print("Min:", min(samples))
print("Max:", max(samples))

mean = sum(samples) / len(samples)

rms = (
    sum(
        float(x) * float(x)
        for x in samples
    ) / len(samples)
) ** 0.5

print("Mean:", mean)
print("RMS:", rms)


# ------------------------------------------------------------
# Save raw PCM16
# ------------------------------------------------------------

with open("esp32_capture.raw", "wb") as f:

    for sample in samples:

        f.write(
            struct.pack("<h", sample)
        )


# ------------------------------------------------------------
# Save text
# ------------------------------------------------------------

with open("esp32_capture.txt", "w") as f:

    for sample in samples:
        f.write(f"{sample}\n")


print()
print("Saved:")
print("  esp32_capture.raw")
print("  esp32_capture.txt")