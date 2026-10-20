import os
import subprocess
import glob

sources = glob.glob("../pluto_proto/managed_components/espressif__esp-tflite-micro/tensorflow/**/*.cc", recursive=True) + \
          glob.glob("../pluto_proto/managed_components/espressif__esp-tflite-micro/tensorflow/**/*.c", recursive=True)

valid_sources = []
for s in sources:
    if "examples/" in s or "test" in s or "esp32" in s or "riscv" in s or "opt.c" in s or "esp_nn" in s or "/esp/" in s or "mock_micro_graph.cc" in s:
        continue
    valid_sources.append(s)

compile_cmd = [
    "g++", "-std=c++17", "-O3", "-DTF_LITE_STATIC_MEMORY",
    "-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics",
    "-Wno-error",
    "-I../pluto_proto/managed_components/espressif__esp-tflite-micro/",
    "-I../pluto_proto/managed_components/espressif__esp-tflite-micro/third_party/flatbuffers/include/",
    "-I../pluto_proto/managed_components/espressif__esp-tflite-micro/third_party/gemmlowp/",
    "-I../pluto_proto/managed_components/espressif__esp-tflite-micro/third_party/ruy/",
    "measure_arena.cpp"
] + valid_sources + ["-o", "measure_arena"]

print(f"Compiling {len(valid_sources)} files...")
subprocess.run(compile_cmd, check=True)
