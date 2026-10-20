from pathlib import Path

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------

MODEL = Path("ds_cnn_int8.tflite")
OUTPUT = Path("main/ds_cnn_int8_model.h")


# --------------------------------------------------
# READ TFLITE MODEL
# --------------------------------------------------

data = MODEL.read_bytes()


# --------------------------------------------------
# CREATE C HEADER
# --------------------------------------------------

with open(OUTPUT, "w") as f:

    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n")
    f.write("#include <stddef.h>\n\n")

    f.write(
        "static const uint8_t ds_cnn_int8_model[] "
        "__attribute__((aligned(16))) = {\n"
    )

    # Write 12 bytes per line
    for i, byte in enumerate(data):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02X}")

        if i != len(data) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    # Finish final line
    if len(data) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")

    f.write(
        f"static const size_t ds_cnn_int8_model_len = {len(data)};\n"
    )


# --------------------------------------------------
# RESULT
# --------------------------------------------------

print("=" * 50)
print("TFLITE MODEL CONVERSION COMPLETE")
print("=" * 50)
print(f"Model size : {len(data)} bytes")
print(f"Input      : {MODEL}")
print(f"Output     : {OUTPUT}")
print("=" * 50)