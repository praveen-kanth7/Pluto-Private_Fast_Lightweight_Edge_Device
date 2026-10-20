#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_allocator.h"

int main() {
    std::ifstream file("tiny_ds_cnn_int8.tflite", std::ios::binary | std::ios::ate);
    if (!file) { std::cerr << "Cannot open model\n"; return 1; }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(size);
    if (!file.read((char*)model_data.data(), size)) { return 1; }
    
    const tflite::Model* model = tflite::GetModel(model_data.data());

    tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();

    std::cout << "Model file size: " << size << " bytes (" << size / 1024.0 << " KB)\n";
    
    // Print tensors and operators
    auto subgraphs = model->subgraphs();
    if (subgraphs && subgraphs->size() > 0) {
        auto subgraph = (*subgraphs)[0];
        std::cout << "Number of tensors: " << subgraph->tensors()->size() << "\n";
        
        auto operators = subgraph->operators();
        std::cout << "Operators: ";
        if (operators) {
            std::cout << operators->size() << " operations\n";
        } else {
            std::cout << "0\n";
        }
    }
    
    // Test with a large enough arena first to print exact head/tail info
    constexpr size_t kInitialArenaSize = 128 * 1024;
    std::vector<uint8_t> initial_arena(kInitialArenaSize + 16);
    uint8_t* arena_ptr = (uint8_t*)(((uintptr_t)initial_arena.data() + 15) & ~15);
    
    {
        tflite::RecordingMicroInterpreter interpreter(model, resolver, arena_ptr, kInitialArenaSize);
        if (interpreter.AllocateTensors() == kTfLiteOk) {
            std::cout << "Initial AllocateTensors SUCCESS (128 KB)\n";
            interpreter.GetMicroAllocator().PrintAllocations();
            
            TfLiteTensor* input = interpreter.input(0);
            TfLiteTensor* output = interpreter.output(0);
            
            std::cout << "Input shape: [";
            for (int i = 0; i < input->dims->size; ++i) {
                std::cout << input->dims->data[i] << (i == input->dims->size - 1 ? "" : ", ");
            }
            std::cout << "]\n";
            
            std::cout << "Output shape: [";
            for (int i = 0; i < output->dims->size; ++i) {
                std::cout << output->dims->data[i] << (i == output->dims->size - 1 ? "" : ", ");
            }
            std::cout << "]\n";
            
            std::cout << "Input type: " << TfLiteTypeGetName(input->type) << "\n";
            std::cout << "Output type: " << TfLiteTypeGetName(output->type) << "\n";
        }
    }
    
    // Binary search for smallest tensor arena size
    size_t min_arena = 0;
    size_t max_arena = 128 * 1024;
    size_t lowest_success = max_arena;
    
    while (min_arena <= max_arena) {
        size_t mid_arena = min_arena + (max_arena - min_arena) / 2;
        
        std::vector<uint8_t> test_arena(mid_arena + 16);
        uint8_t* aligned_ptr = (uint8_t*)(((uintptr_t)test_arena.data() + 15) & ~15);
        
        tflite::MicroInterpreter test_interpreter(model, resolver, aligned_ptr, mid_arena);
        if (test_interpreter.AllocateTensors() == kTfLiteOk) {
            lowest_success = mid_arena;
            max_arena = mid_arena - 1;
        } else {
            min_arena = mid_arena + 1;
        }
    }
    
    std::cout << "Smallest tested failing arena: " << lowest_success - 1 << " bytes\n";
    std::cout << "Minimum tested successful arena: " << lowest_success << " bytes (" << lowest_success / 1024.0 << " KB)\n";

    return 0;
}
