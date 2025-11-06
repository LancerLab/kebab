#include "kebab/utils/benchmark.h"
#include "kebab/cute/elementwise_add.h"
#include "kebab/config/config_parser.h"
#include "kebab/cuda/cuda_elementwise_add.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include <iomanip>

using namespace kebab::benchmark;
using namespace kebab::config;

/**
 * @brief Single-run elementwise_add kernel for profiling
 * 
 * This binary runs the elementwise_add kernel exactly once (no loops, no warmup).
 * Designed for use with Nsight Compute profiling.
 * 
 * Usage: ./runonce_elementwise_add
 * 
 * Configuration is read from config.yaml
 */
template<typename T>
void runOnceElementwiseAdd(const ConfigParser& config) {
    std::string type_name = std::is_same_v<T, float> ? "float" : "half";
    std::cout << "\n========================================" << std::endl;
    std::cout << "Single-Run Elementwise Add Kernel (" << type_name << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get configuration
    std::string impl = config.getOperatorImpl("elementwise_add");
    auto batch_sizes = config.getOperatorSizes("elementwise_add");
    bool verbose = config.getOperatorVerbose("elementwise_add");
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Implementation: " << impl << std::endl;
    std::cout << "  Batch size: " << batch_sizes[0] << std::endl;
    std::cout << "  Precision: " << type_name << std::endl;
    std::cout << "  Note: Running kernel ONCE (no loops, no warmup)" << std::endl;
    std::cout << std::endl;
    
    // Use first batch size
    int N = batch_sizes[0];
    
    std::cout << "Array Configuration:" << std::endl;
    std::cout << "  N=" << N << " elements" << std::endl;
    std::cout << "  Memory per array: " << (N * sizeof(T) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << std::endl;
    
    // Allocate host memory
    std::vector<T> h_A(N), h_B(N), h_C(N);
    
    // Initialize arrays
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < N; ++i) {
        if constexpr (sizeof(T) == 4) {  // float
            h_A[i] = static_cast<T>(dist(gen));
            h_B[i] = static_cast<T>(dist(gen));
        } else {  // half
            h_A[i] = __float2half(dist(gen));
            h_B[i] = __float2half(dist(gen));
        }
        h_C[i] = T{0};
    }
    
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(T)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, N * sizeof(T)));
    
    // Synchronize before profiling
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "Running kernel once..." << std::endl;

    // Run kernel exactly once (no loops, no warmup)
    if (impl == "cute") {
        kebab::cute::elementwise_add(d_A, d_B, d_C, N);
    } else {
        baseline::elementwise_add(d_A, d_B, d_C, N);
    }

    // Synchronize after kernel
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "Kernel execution complete!" << std::endl;
    std::cout << "Ready for profiling analysis." << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Single-Run Elementwise Add Profiling Binary" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // Load configuration (singleton)
        auto& config = ConfigParser::getInstance("config.yaml");

        // Check if elementwise_add is enabled
        if (!config.isOperatorEnabled("elementwise_add")) {
            std::cerr << "ERROR: Elementwise_add operator is not enabled in config.yaml" << std::endl;
            return 1;
        }

        // Set GPU device
        int gpu_id = config.getOperatorGpuId("elementwise_add");
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (gpu_id >= 0 && gpu_id < device_count) {
            CUDA_CHECK(cudaSetDevice(gpu_id));
        } else if (gpu_id >= device_count) {
            std::cerr << "Warning: GPU " << gpu_id << " not available (only " << device_count
                      << " GPUs detected). Using GPU 0." << std::endl;
            CUDA_CHECK(cudaSetDevice(0));
            gpu_id = 0;
        } else {
            CUDA_CHECK(cudaGetDevice(&gpu_id));
        }

        std::cout << "Configuration:" << std::endl;
        std::cout << "  GPU ID: " << gpu_id << std::endl;

        // Get GPU properties
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "\nGPU Information:" << std::endl;
        std::cout << "  Device: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;

        bool has_tensor_cores = (prop.major >= 7);
        std::cout << "  Tensor Cores: " << (has_tensor_cores ? "Supported" : "Not Supported") << std::endl;

        // Get precisions from config
        auto precisions = config.getOperatorPrecisions("elementwise_add");

        // Run for each precision
        for (const auto& precision : precisions) {
            if (precision == "float32" || precision == "float") {
                runOnceElementwiseAdd<float>(config);
            } else if (precision == "float16" || precision == "half") {
                if (has_tensor_cores) {
                    runOnceElementwiseAdd<__half>(config);
                } else {
                    std::cout << "\nSkipping half precision (requires Tensor Core support)" << std::endl;
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

