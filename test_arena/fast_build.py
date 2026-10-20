import os
import subprocess
import glob
import multiprocessing

sources = glob.glob("../pluto_proto/managed_components/espressif__esp-tflite-micro/tensorflow/**/*.cc", recursive=True) + \
          glob.glob("../pluto_proto/managed_components/espressif__esp-tflite-micro/tensorflow/**/*.c", recursive=True)

valid_sources = ["measure_arena.cpp"]
for s in sources:
    if "examples/" in s or "test" in s or "esp32" in s or "riscv" in s or "opt.c" in s or "esp_nn" in s or "/esp/" in s or "mock_micro_graph.cc" in s or "kernel_runner.cc" in s:
        continue
    valid_sources.append(s)

with open("Makefile", "w") as f:
    f.write("CXX = g++\n")
    f.write("CC = gcc\n")
    f.write("CXXFLAGS = -std=c++17 -O3 -DTF_LITE_STATIC_MEMORY -fno-exceptions -fno-rtti -fno-threadsafe-statics -Wno-error ")
    f.write("-I../pluto_proto/managed_components/espressif__esp-tflite-micro/ ")
    f.write("-I../pluto_proto/managed_components/espressif__esp-tflite-micro/third_party/flatbuffers/include/ ")
    f.write("-I../pluto_proto/managed_components/espressif__esp-tflite-micro/third_party/gemmlowp/ ")
    f.write("-I../pluto_proto/managed_components/espressif__esp-tflite-micro/third_party/ruy/\n\n")
    
    objs = []
    for s in valid_sources:
        obj = s.replace("../", "").replace("/", "_") + ".o"
        objs.append(obj)
        if s.endswith(".c"):
            f.write(f"{obj}: {s}\n\t$(CC) $(CXXFLAGS) -c $< -o $@\n\n")
        else:
            f.write(f"{obj}: {s}\n\t$(CXX) $(CXXFLAGS) -c $< -o $@\n\n")
            
    f.write(f"measure_arena: {' '.join(objs)}\n\t$(CXX) $(CXXFLAGS) $^ -o $@\n\n")

print(f"Generated Makefile with {len(valid_sources)} files. Compiling with {multiprocessing.cpu_count()} threads...")
subprocess.run(["make", f"-j{multiprocessing.cpu_count()}", "measure_arena"], check=True)
