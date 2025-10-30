# CuTeKernelLib Developer Guide

This guide provides step-by-step instructions for extending CuTeKernelLib with new operators, baselines, and benchmarks.

## Table of Contents

1. [Overview](#overview)
2. [Adding a New Operator](#adding-a-new-operator)
3. [Code Templates](#code-templates)
4. [Makefile Integration](#makefile-integration)
5. [Configuration Extension](#configuration-extension)
6. [Testing and Verification](#testing-and-verification)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)

## Overview

CuTeKernelLib follows a modular architecture where each operator consists of:

1. **CuTe Implementation**: Template-based GPU kernel using CUTLASS CuTe
2. **CUDA Baseline**: Hand-optimized CUDA reference implementation
3. **Benchmark Driver**: Performance measurement and comparison program
4. **Unit Tests**: Correctness verification tests
5. **Makefile Integration**: Build system integration
6. **Configuration**: Parameters in `config.yaml`

## Adding a New Operator

### Step 1: Plan Your Operator

Before implementing, define:
- **Operator name**: Use snake_case (e.g., `matrix_multiply`, `conv2d`)
- **Input/output shapes**: Matrix dimensions, tensor shapes
- **Data types**: float32, float16, int8, etc.
- **Performance targets**: Target GFLOPS, memory bandwidth
- **Hardware features**: Tensor Cores, shared memory, async copy

### Step 2: Create Operator Header

Create `include/cutekernellib/operators/<operator_name>.h`:

```cpp
#pragma once
#include <cuda_runtime.h>

namespace cutekernellib {

// Error checking macro (reuse from existing operators)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif

// Public API function
template<typename T>
void your_operator(const T* input_a, const T* input_b, T* output,
                   int param1, int param2, int param3,
                   cudaStream_t stream = 0);

} // namespace cutekernellib
```

### Step 3: Implement CuTe Operator

Create `src/operators/<operator_name>.cu`:

```cpp
#include "cutekernellib/operators/<operator_name>.h"
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>  // Or relevant CuTe algorithms

namespace cutekernellib {

template<typename T>
__global__ void your_operator_kernel(const T* input_a, const T* input_b, T* output,
                                     int param1, int param2, int param3) {
    using namespace cute;
    
    // 1. Define tensor layouts
    auto gA = make_tensor(make_gmem_ptr(input_a), make_shape(param1, param2));
    auto gB = make_tensor(make_gmem_ptr(input_b), make_shape(param2, param3));
    auto gC = make_tensor(make_gmem_ptr(output), make_shape(param1, param3));
    
    // 2. Create tiled MMA (for compute-intensive operators)
    // Example for Tensor Core usage:
    // auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
    
    // 3. Thread and block mapping
    auto thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 4. Implement your algorithm using CuTe primitives
    // This is operator-specific - see existing operators for patterns
    
    // 5. Use CuTe copy operations for memory transfers
    // auto src_tensor = local_tile(gA, ...);
    // auto dst_tensor = local_tile(shared_memory, ...);
    // copy(src_tensor, dst_tensor);
}

template<typename T>
void your_operator(const T* input_a, const T* input_b, T* output,
                   int param1, int param2, int param3, cudaStream_t stream) {
    // Input validation
    if (input_a == nullptr || input_b == nullptr || output == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to your_operator\n");
        return;
    }
    if (param1 <= 0 || param2 <= 0 || param3 <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: %d, %d, %d\n", param1, param2, param3);
        return;
    }
    
    // Kernel launch configuration
    dim3 block(256);  // Adjust based on operator requirements
    dim3 grid((param1 * param3 + block.x - 1) / block.x);
    
    // Launch kernel
    your_operator_kernel<<<grid, block, 0, stream>>>(
        input_a, input_b, output, param1, param2, param3);
    
    // Error checking
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations
template void your_operator<float>(const float*, const float*, float*, 
                                   int, int, int, cudaStream_t);
template void your_operator<__half>(const __half*, const __half*, __half*, 
                                    int, int, int, cudaStream_t);

} // namespace cutekernellib
```

### Step 4: Create CUDA Baseline

Create `baselines/cuda/cuda_<operator_name>.cu`:

```cpp
#include "cuda_<operator_name>.h"
#include <cuda_runtime.h>

namespace baseline {

template<typename T>
__global__ void optimized_your_operator_kernel(const T* input_a, const T* input_b, T* output,
                                               int param1, int param2, int param3) {
    // Hand-optimized CUDA implementation
    // Use techniques like:
    // - Shared memory tiling
    // - Warp-level primitives (wmma for Tensor Cores)
    // - Register blocking
    // - Memory coalescing
    // - Loop unrolling
    
    __shared__ T shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    T result = 0;
    
    // Tiled computation loop
    for (int tile = 0; tile < (param2 + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile into shared memory
        if (row < param1 && tile * TILE_SIZE + tx < param2)
            shared_a[ty][tx] = input_a[row * param2 + tile * TILE_SIZE + tx];
        else
            shared_a[ty][tx] = 0;
            
        if (col < param3 && tile * TILE_SIZE + ty < param2)
            shared_b[ty][tx] = input_b[(tile * TILE_SIZE + ty) * param3 + col];
        else
            shared_b[ty][tx] = 0;
            
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            result += shared_a[ty][k] * shared_b[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < param1 && col < param3) {
        output[row * param3 + col] = result;
    }
}

template<typename T>
void your_operator(const T* input_a, const T* input_b, T* output,
                   int param1, int param2, int param3) {
    const int TILE_SIZE = 32;  // Adjust based on operator
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((param3 + TILE_SIZE - 1) / TILE_SIZE, 
              (param1 + TILE_SIZE - 1) / TILE_SIZE);
    
    optimized_your_operator_kernel<<<grid, block>>>(
        input_a, input_b, output, param1, param2, param3);
}

// Explicit instantiations
template void your_operator<float>(const float*, const float*, float*, int, int, int);
template void your_operator<__half>(const __half*, const __half*, __half*, int, int, int);

} // namespace baseline
```

Create corresponding header `baselines/cuda/cuda_<operator_name>.h`:

```cpp
#pragma once
#include <cuda_runtime.h>

namespace baseline {

template<typename T>
void your_operator(const T* input_a, const T* input_b, T* output,
                   int param1, int param2, int param3);

} // namespace baseline
```

### Step 5: Create Benchmark Driver

Create `benchmarks/bench_<operator_name>.cu`:

```cpp
#include "cutekernellib/operators/<operator_name>.h"
#include "baselines/cuda/cuda_<operator_name>.h"
#include "benchmark_runner.h"
#include "../src/config/config_parser.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

int main(int argc, char** argv) {
    try {
        // Load configuration
        auto& config = ConfigParser::getInstance("config.yaml");
        
        int warmup_runs = config.getWarmupRuns();
        int measurement_runs = config.getMeasurementRuns();
        auto batch_sizes = config.getBatchSizes();
        
        // Initialize benchmark runner
        BenchmarkRunner runner(warmup_runs, measurement_runs);
        std::vector<BenchmarkResult> results;
        
        std::cout << "========================================\n";
        std::cout << "Your Operator Benchmark\n";
        std::cout << "========================================\n";
        std::cout << "Warmup runs: " << warmup_runs << "\n";
        std::cout << "Measurement runs: " << measurement_runs << "\n";
        std::cout << "Batch sizes: ";
        for (size_t i = 0; i < batch_sizes.size(); ++i) {
            std::cout << batch_sizes[i];
            if (i < batch_sizes.size() - 1) std::cout << ", ";
        }
        std::cout << "\n\n";
        
        // Random number generator for test data
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (int size : batch_sizes) {
            // Define problem dimensions (adjust for your operator)
            int param1 = size;
            int param2 = size;
            int param3 = size;
            
            std::cout << "Testing size: " << size << "x" << size << "x" << size << "\n";
            
            // Allocate host memory
            size_t size_a = param1 * param2 * sizeof(float);
            size_t size_b = param2 * param3 * sizeof(float);
            size_t size_c = param1 * param3 * sizeof(float);
            
            std::vector<float> h_a(param1 * param2);
            std::vector<float> h_b(param2 * param3);
            
            // Initialize with random data
            for (auto& val : h_a) val = dist(gen);
            for (auto& val : h_b) val = dist(gen);
            
            // Allocate device memory
            float *d_a, *d_b, *d_c_cute, *d_c_cuda;
            cudaMalloc(&d_a, size_a);
            cudaMalloc(&d_b, size_b);
            cudaMalloc(&d_c_cute, size_c);
            cudaMalloc(&d_c_cuda, size_c);
            
            // Copy data to device
            cudaMemcpy(d_a, h_a.data(), size_a, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice);
            
            // Benchmark CuTe implementation
            auto cute_kernel = [&]() {
                cutekernellib::your_operator(d_a, d_b, d_c_cute, param1, param2, param3);
            };
            float cute_latency = runner.measureLatency(cute_kernel);
            
            // Benchmark CUDA baseline
            auto cuda_kernel = [&]() {
                baseline::your_operator(d_a, d_b, d_c_cuda, param1, param2, param3);
            };
            float cuda_latency = runner.measureLatency(cuda_kernel);
            
            // Calculate performance metrics
            // Adjust FLOP count for your specific operator
            double flops = 2.0 * param1 * param2 * param3;  // Example for GEMM
            float cute_gflops = (flops / 1e9) / (cute_latency / 1e3);
            float cuda_gflops = (flops / 1e9) / (cuda_latency / 1e3);
            float speedup = cuda_latency / cute_latency;
            
            // Store results
            results.push_back({
                "YourOperator", "CuTe", size, cute_latency, cute_gflops, speedup
            });
            results.push_back({
                "YourOperator", "CUDA", size, cuda_latency, cuda_gflops, 1.0f
            });
            
            std::cout << "  CuTe: " << std::fixed << std::setprecision(3) 
                      << cute_latency << " ms, " << cute_gflops << " GFLOPS\n";
            std::cout << "  CUDA: " << cuda_latency << " ms, " << cuda_gflops << " GFLOPS\n";
            std::cout << "  Speedup: " << speedup << "x\n\n";
            
            // Cleanup
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c_cute);
            cudaFree(d_c_cuda);
        }
        
        // Save results to CSV
        std::ofstream csv("bench_results/your_operator_results.csv");
        csv << "Operator,Variant,BatchSize,Latency(ms),Throughput(GFLOPS),Speedup\n";
        for (const auto& r : results) {
            csv << r.operator_name << "," << r.variant << "," 
                << r.batch_size << "," << r.latency_ms << "," 
                << r.throughput_gflops << "," << r.speedup_ratio << "\n";
        }
        csv.close();
        
        std::cout << "Results saved to: bench_results/your_operator_results.csv\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### Step 6: Create Unit Tests

Create `tests/test_<operator_name>.cpp`:

```cpp
#include "cutekernellib/operators/<operator_name>.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <cassert>

bool test_correctness() {
    const int param1 = 64, param2 = 64, param3 = 64;
    
    // Generate test data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> h_a(param1 * param2);
    std::vector<float> h_b(param2 * param3);
    std::vector<float> h_c_gpu(param1 * param3);
    std::vector<float> h_c_cpu(param1 * param3, 0.0f);
    
    for (auto& val : h_a) val = dist(gen);
    for (auto& val : h_b) val = dist(gen);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, param1 * param2 * sizeof(float));
    cudaMalloc(&d_b, param2 * param3 * sizeof(float));
    cudaMalloc(&d_c, param1 * param3 * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_a, h_a.data(), param1 * param2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), param2 * param3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Run GPU kernel
    cutekernellib::your_operator(d_a, d_b, d_c, param1, param2, param3);
    
    // Copy result back
    cudaMemcpy(h_c_gpu.data(), d_c, param1 * param3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute CPU reference (adjust for your operator)
    // Example for matrix multiplication:
    for (int i = 0; i < param1; ++i) {
        for (int j = 0; j < param3; ++j) {
            for (int k = 0; k < param2; ++k) {
                h_c_cpu[i * param3 + j] += h_a[i * param2 + k] * h_b[k * param3 + j];
            }
        }
    }
    
    // Compare results
    const float tolerance = 1e-3f;
    for (int i = 0; i < param1 * param3; ++i) {
        float diff = std::abs(h_c_gpu[i] - h_c_cpu[i]);
        if (diff > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_c_gpu[i] 
                      << ", CPU=" << h_c_cpu[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return true;
}

bool test_error_handling() {
    // Test null pointer handling
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 1024 * sizeof(float));
    cudaMalloc(&d_b, 1024 * sizeof(float));
    cudaMalloc(&d_c, 1024 * sizeof(float));
    
    // These should not crash (should handle gracefully)
    cutekernellib::your_operator<float>(nullptr, d_b, d_c, 32, 32, 32);
    cutekernellib::your_operator<float>(d_a, nullptr, d_c, 32, 32, 32);
    cutekernellib::your_operator<float>(d_a, d_b, nullptr, 32, 32, 32);
    
    // Test invalid dimensions
    cutekernellib::your_operator<float>(d_a, d_b, d_c, 0, 32, 32);
    cutekernellib::your_operator<float>(d_a, d_b, d_c, 32, 0, 32);
    cutekernellib::your_operator<float>(d_a, d_b, d_c, 32, 32, 0);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return true;
}

int main() {
    std::cout << "Testing YourOperator correctness..." << std::endl;
    
    if (!test_correctness()) {
        std::cerr << "Correctness test failed!" << std::endl;
        return 1;
    }
    std::cout << "✓ Correctness test passed" << std::endl;
    
    if (!test_error_handling()) {
        std::cerr << "Error handling test failed!" << std::endl;
        return 1;
    }
    std::cout << "✓ Error handling test passed" << std::endl;
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
```

## Makefile Integration

### Step 1: Add Operator to Lists

Edit the `Makefile` and add your operator to the appropriate lists:

```makefile
# Add to OPERATORS list
OPERATORS := elementwise_add your_operator

# Add to BASELINES list  
BASELINES := cuda_elementwise_add cuda_your_operator
```

### Step 2: Add Compilation Rules

The existing Makefile uses pattern rules, so your operator should compile automatically. However, if you need custom compilation flags, add:

```makefile
# Custom compilation for your operator (if needed)
$(BUILD_DIR)/your_operator.o: $(SRC_DIR)/operators/your_operator.cu
	@echo "Compiling your_operator with custom flags..."
	$(NVCC) $(NVCC_FLAGS) -DCUSTOM_FLAG=1 -c $< -o $@
```

### Step 3: Add Benchmark Target

Add benchmark target (usually automatic with pattern rules):

```makefile
# This should work automatically, but you can add custom rules if needed
bench-your_operator: $(BUILD_DIR)/bench_your_operator
	@echo "Running YourOperator benchmark..."
	@$(BUILD_DIR)/bench_your_operator
```

### Step 4: Add Test Target

Add to the test target:

```makefile
test: test-config test-elementwise_add test-your_operator test-cuda-baseline-your_operator

test-your_operator: $(BUILD_DIR)/test_your_operator
	@echo "Running YourOperator test..."
	@$(BUILD_DIR)/test_your_operator

$(BUILD_DIR)/test_your_operator: tests/test_your_operator.cpp $(BUILD_DIR)/libcutekernellib.a
	$(NVCC) $(NVCC_FLAGS) $< -L$(BUILD_DIR) -lcutekernellib -o $@
```

## Configuration Extension

### Add Operator Parameters

Edit `config.yaml` to add your operator configuration:

```yaml
operators:
  your_operator:
    enabled: true
    tile_sizes: [16, 32, 64]        # Operator-specific parameters
    matrix_sizes: [256, 512, 1024]  # Test sizes
    data_types: [float32, float16]  # Supported types
    custom_param: 42                # Any custom parameters
```

### Update ConfigParser (if needed)

If you need custom configuration parsing, extend `src/config/config_parser.cpp`:

```cpp
// Add to ConfigParser class
std::vector<int> getYourOperatorTileSizes() const {
    if (config_["operators"]["your_operator"]["tile_sizes"]) {
        return config_["operators"]["your_operator"]["tile_sizes"].as<std::vector<int>>();
    }
    return {16, 32, 64};  // Default values
}

int getYourOperatorCustomParam() const {
    if (config_["operators"]["your_operator"]["custom_param"]) {
        return config_["operators"]["your_operator"]["custom_param"].as<int>();
    }
    return 42;  // Default value
}
```

## Testing and Verification

### Step 1: Compile and Test

```bash
# Build everything
make build

# Run unit tests
make test-your_operator

# Run benchmark
make bench-your_operator

# Profile with ncu
make tune-your_operator
```

### Step 2: Verify Correctness

1. **Unit tests pass**: Ensure your implementation matches CPU reference
2. **No CUDA errors**: Check for runtime errors
3. **Performance reasonable**: Compare against baseline and expectations
4. **Memory safety**: Use cuda-memcheck if available

### Step 3: Performance Validation

```bash
# Run comprehensive benchmark
make bench-your_operator

# Check performance metrics
cat bench_results/your_operator_results.csv

# Profile for optimization opportunities  
make tune-your_operator
ncu-ui profiling/your_operator_profile.ncu-rep
```

## Tensor Core Programming with CuTe

### Overview

Tensor Cores are specialized matrix multiplication units available on modern NVIDIA GPUs (Volta, Turing, Ampere, Ada Lovelace, Hopper). CuTe provides high-level abstractions called MMA atoms that make Tensor Core programming more accessible while maintaining maximum performance.

### Supported MMA Atoms by Architecture

#### Ampere (SM80, SM86) - RTX 30xx, A100
```cpp
// 16x8x16 Tensor Core operation: D = A * B + C
// A: 16x16 (float16), B: 16x8 (float16), C/D: 16x8 (float32)
using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;

// Alternative configurations:
SM80_16x8x16_F32F16F16F32_TN  // A: row-major, B: col-major
SM80_16x8x16_F32F16F16F32_NT  // A: col-major, B: row-major
SM80_16x8x16_F16F16F16F16_TN  // All half precision
```

#### Ada Lovelace (SM89) - RTX 40xx
```cpp
// Enhanced Tensor Cores with sparsity support
using MMA_Atom = SM89_16x8x16_F32F16F16F32_TN;
// Also supports structured sparsity (2:4 sparse patterns)
```

#### Hopper (SM90) - H100
```cpp
// Larger Tensor Core operations
using MMA_Atom = SM90_64x128x16_F32F16F16F32_TN;
using MMA_Atom = SM90_64x64x16_F32F16F16F32_TN;
```

### Basic Tensor Core Usage Pattern

Here's the fundamental pattern for using Tensor Cores in CuTe:

```cpp
template<typename T>
__global__ void tensor_core_gemm_kernel(
    const T* A_ptr, const T* B_ptr, T* C_ptr,
    int M, int N, int K)
{
    using namespace cute;
    
    // 1. Define MMA atom for your target architecture
    using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_2,_2,_1>>>;
    
    // 2. Create global tensors
    auto gA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K));
    auto gB = make_tensor(make_gmem_ptr(B_ptr), make_shape(K, N));
    auto gC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N));
    
    // 3. Create tiled MMA
    TiledMMA tiled_mma;
    
    // 4. Get thread-local partitions
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA);  // Thread's A partition
    auto tBgB = thr_mma.partition_B(gB);  // Thread's B partition  
    auto tCgC = thr_mma.partition_C(gC);  // Thread's C partition
    
    // 5. Create accumulators
    auto tCrC = thr_mma.partition_C(make_tensor<float>(make_shape(size(tCgC))));
    clear(tCrC);  // Initialize to zero
    
    // 6. Main computation loop
    for (int k_tile = 0; k_tile < size<1>(tAgA); ++k_tile) {
        // Load A and B fragments
        auto tArA = thr_mma.make_fragment_A(tAgA(_, k_tile));
        auto tBrB = thr_mma.make_fragment_B(tBgB(k_tile, _));
        
        // Perform Tensor Core operation
        gemm(tiled_mma, tArA, tBrB, tCrC);
    }
    
    // 7. Store result
    copy(tCrC, tCgC);
}
```

### Advanced Tensor Core Patterns

#### 1. Shared Memory Tiling with Tensor Cores

```cpp
template<typename T>
__global__ void advanced_tensor_core_gemm(
    const T* A_ptr, const T* B_ptr, T* C_ptr,
    int M, int N, int K)
{
    using namespace cute;
    
    // Define tile sizes
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128; 
    constexpr int TILE_K = 32;
    
    // Shared memory
    __shared__ T smem_A[TILE_M][TILE_K];
    __shared__ T smem_B[TILE_K][TILE_N];
    
    // MMA configuration
    using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_8,_4,_1>>>;  // 8x4 thread blocks
    
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    
    // Global tensors
    auto gA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K));
    auto gB = make_tensor(make_gmem_ptr(B_ptr), make_shape(K, N));
    auto gC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N));
    
    // Shared memory tensors
    auto sA = make_tensor(make_smem_ptr(smem_A), make_shape(TILE_M, TILE_K));
    auto sB = make_tensor(make_smem_ptr(smem_B), make_shape(TILE_K, TILE_N));
    
    // Thread partitions
    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);
    
    auto tAsA = thr_mma.partition_A(sA);
    auto tBsB = thr_mma.partition_B(sB);
    
    // Accumulator
    auto tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; ++k_tile) {
        // Copy global to shared memory
        copy(tAgA(_, k_tile), tAsA);
        copy(tBgB(k_tile, _), tBsB);
        __syncthreads();
        
        // Create fragments from shared memory
        auto tArA = thr_mma.make_fragment_A(tAsA);
        auto tBrB = thr_mma.make_fragment_B(tBsB);
        
        // Tensor Core computation
        gemm(tiled_mma, tArA, tBrB, tCrC);
        __syncthreads();
    }
    
    // Store final result
    copy(tCrC, tCgC);
}
```

#### 2. Software Pipelining with Async Copy

```cpp
template<typename T>
__global__ void pipelined_tensor_core_gemm(
    const T* A_ptr, const T* B_ptr, T* C_ptr,
    int M, int N, int K)
{
    using namespace cute;
    
    // Double-buffered shared memory
    __shared__ T smem_A[2][TILE_M][TILE_K];
    __shared__ T smem_B[2][TILE_K][TILE_N];
    
    using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_8,_4,_1>>>;
    
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    
    // Create tensors for both buffers
    auto sA0 = make_tensor(make_smem_ptr(smem_A[0]), make_shape(TILE_M, TILE_K));
    auto sA1 = make_tensor(make_smem_ptr(smem_A[1]), make_shape(TILE_M, TILE_K));
    auto sB0 = make_tensor(make_smem_ptr(smem_B[0]), make_shape(TILE_K, TILE_N));
    auto sB1 = make_tensor(make_smem_ptr(smem_B[1]), make_shape(TILE_K, TILE_N));
    
    // Pipeline loop
    int write_stage = 0;
    int read_stage = 1;
    
    // Prefetch first tile
    copy_async(tAgA(_, 0), thr_mma.partition_A(sA0));
    copy_async(tBgB(0, _), thr_mma.partition_B(sB0));
    cp_async_wait<0>();
    __syncthreads();
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Swap buffers
        write_stage = 1 - write_stage;
        read_stage = 1 - read_stage;
        
        // Async copy next tile (if available)
        if (k_tile + 1 < num_k_tiles) {
            auto sA_write = (write_stage == 0) ? sA0 : sA1;
            auto sB_write = (write_stage == 0) ? sB0 : sB1;
            
            copy_async(tAgA(_, k_tile + 1), thr_mma.partition_A(sA_write));
            copy_async(tBgB(k_tile + 1, _), thr_mma.partition_B(sB_write));
        }
        
        // Compute with current tile
        auto sA_read = (read_stage == 0) ? sA0 : sA1;
        auto sB_read = (read_stage == 0) ? sB0 : sB1;
        
        auto tArA = thr_mma.make_fragment_A(thr_mma.partition_A(sA_read));
        auto tBrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB_read));
        
        // Tensor Core operation
        gemm(tiled_mma, tArA, tBrB, tCrC);
        
        // Wait for async copy to complete
        cp_async_wait<0>();
        __syncthreads();
    }
    
    // Store result
    copy(tCrC, tCgC);
}
```

### MMA Atom Selection Guide

#### Choosing the Right MMA Atom

1. **Data Type Considerations**:
   ```cpp
   // Mixed precision (recommended for performance)
   SM80_16x8x16_F32F16F16F32_TN  // A,B: half, C,D: float
   
   // Full precision
   SM80_16x8x16_F32F32F32F32_TN  // All float32
   
   // Half precision throughout
   SM80_16x8x16_F16F16F16F16_TN  // All half
   ```

2. **Matrix Layout**:
   ```cpp
   // TN: A row-major (Transposed=false), B col-major (Transposed=true)
   SM80_16x8x16_F32F16F16F32_TN
   
   // NT: A col-major, B row-major  
   SM80_16x8x16_F32F16F16F32_NT
   
   // NN: Both row-major
   SM80_16x8x16_F32F16F16F32_NN
   ```

3. **Performance Characteristics**:
   - **16x8x16**: Balanced for most workloads
   - **16x8x8**: Higher throughput for smaller K dimensions
   - **8x8x4**: For very small matrices or when register pressure is high

### Tensor Core Best Practices

#### 1. Memory Layout Optimization

```cpp
// Ensure proper alignment for Tensor Core operations
template<typename T>
void prepare_tensor_core_data(T* matrix, int rows, int cols) {
    // Tensor Cores require 16-byte alignment
    assert(reinterpret_cast<uintptr_t>(matrix) % 16 == 0);
    
    // For half precision, ensure even dimensions for vectorized access
    if constexpr (std::is_same_v<T, __half>) {
        assert(cols % 2 == 0);  // Enable half2 vectorization
    }
}
```

#### 2. Thread Block Configuration

```cpp
// Optimal thread block sizes for Tensor Cores
template<typename MMA_Atom>
constexpr auto get_optimal_thread_layout() {
    if constexpr (std::is_same_v<MMA_Atom, SM80_16x8x16_F32F16F16F32_TN>) {
        // 32 threads per warp, 4 warps per block = 128 threads
        return Layout<Shape<_8, _4, _1>>{};  // 8x4 thread arrangement
    }
    // Add other MMA atom configurations...
}
```

#### 3. Shared Memory Bank Conflict Avoidance

```cpp
// Pad shared memory to avoid bank conflicts
template<int TILE_M, int TILE_K>
struct SharedMemoryLayout {
    // Add padding to avoid 32-way bank conflicts
    static constexpr int PADDING = (TILE_K % 32 == 0) ? 8 : 0;
    using type = T[TILE_M][TILE_K + PADDING];
};

__shared__ typename SharedMemoryLayout<128, 32>::type smem_A;
```

#### 4. Register Pressure Management

```cpp
// Minimize register usage for higher occupancy
template<typename MMA_Atom>
__global__ void __launch_bounds__(256, 2)  // 256 threads, min 2 blocks per SM
optimized_tensor_core_kernel(...) {
    // Use smaller accumulator tiles if register pressure is high
    constexpr int ACC_TILE_M = 64;  // Reduce from 128 if needed
    constexpr int ACC_TILE_N = 64;
    
    // Fragment size affects register usage
    auto tCrC = thr_mma.make_fragment_C(make_shape(ACC_TILE_M, ACC_TILE_N));
}
```

### Complete GEMM Implementation with Tensor Cores

Here's a production-ready GEMM implementation using CuTe and Tensor Cores:

```cpp
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_atom.hpp>

template<typename T>
__global__ void production_gemm_kernel(
    const T* A_ptr, const T* B_ptr, T* C_ptr,
    int M, int N, int K)
{
    using namespace cute;
    
    // Configuration
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 32;
    
    // MMA atom selection based on architecture
    #if __CUDA_ARCH__ >= 800
        using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;
    #elif __CUDA_ARCH__ >= 750
        using MMA_Atom = SM75_16x8x8_F32F16F16F32_TN;
    #else
        #error "Tensor Cores require compute capability 7.5 or higher"
    #endif
    
    // Thread block layout (8x4 = 32 threads per warp, 4 warps = 128 threads)
    using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_8,_4,_1>>>;
    
    // Shared memory with padding
    __shared__ T smem_A[TILE_M][TILE_K + 8];  // +8 padding
    __shared__ T smem_B[TILE_K][TILE_N + 8];  // +8 padding
    
    // Create tensors
    auto gA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K));
    auto gB = make_tensor(make_gmem_ptr(B_ptr), make_shape(K, N));
    auto gC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M, N));
    
    auto sA = make_tensor(make_smem_ptr(smem_A), make_shape(TILE_M, TILE_K));
    auto sB = make_tensor(make_smem_ptr(smem_B), make_shape(TILE_K, TILE_N));
    
    // Tiled MMA
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    
    // Block-level tiling
    auto bA = local_tile(gA, make_shape(TILE_M, TILE_K), make_coord(blockIdx.y, _));
    auto bB = local_tile(gB, make_shape(TILE_K, TILE_N), make_coord(_, blockIdx.x));
    auto bC = local_tile(gC, make_shape(TILE_M, TILE_N), make_coord(blockIdx.y, blockIdx.x));
    
    // Thread partitions
    auto tAgA = thr_mma.partition_A(bA);
    auto tBgB = thr_mma.partition_B(bB);
    auto tCgC = thr_mma.partition_C(bC);
    
    auto tAsA = thr_mma.partition_A(sA);
    auto tBsB = thr_mma.partition_B(sB);
    
    // Accumulator
    auto tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);
    
    // Copy operations for loading data
    auto g2s_tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, T>{}, tiled_mma);
    auto g2s_tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, T>{}, tiled_mma);
    
    auto g2s_thr_copy_A = g2s_tiled_copy_A.get_slice(threadIdx.x);
    auto g2s_thr_copy_B = g2s_tiled_copy_B.get_slice(threadIdx.x);
    
    // Main K loop
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        // Copy global to shared
        copy(g2s_tiled_copy_A, tAgA(_, k_tile), tAsA);
        copy(g2s_tiled_copy_B, tBgB(k_tile, _), tBsB);
        __syncthreads();
        
        // Create fragments from shared memory
        auto tArA = thr_mma.make_fragment_A(tAsA);
        auto tBrB = thr_mma.make_fragment_B(tBsB);
        
        // Tensor Core GEMM
        gemm(tiled_mma, tArA, tBrB, tCrC);
        __syncthreads();
    }
    
    // Store result to global memory
    copy(tCrC, tCgC);
}
```

### Performance Optimization Tips

#### 1. Occupancy Optimization

```cpp
// Check occupancy with different configurations
template<typename Kernel>
void check_occupancy(Kernel kernel) {
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel);
    
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);
    
    printf("Optimal block size: %d\n", block_size);
    printf("Max active blocks per SM: %d\n", max_active_blocks);
}
```

#### 2. Memory Access Pattern Optimization

```cpp
// Ensure coalesced memory access
template<typename Tensor>
void verify_coalesced_access(Tensor& tensor) {
    // CuTe tensors should have stride-1 in the innermost dimension
    static_assert(stride<0>(tensor) == 1, "Innermost dimension must be contiguous");
}
```

#### 3. Tensor Core Utilization Verification

```cpp
// Verify Tensor Core usage in profiling
// Look for these metrics in ncu output:
// - sm__inst_executed_pipe_tensor.sum > 0
// - sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active > 80%
// - smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct > 90%
```

## Performance Optimization

### CuTe Optimization Techniques

1. **Use Tensor Cores**: For compute-intensive operators
   ```cpp
   auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
   ```

2. **Optimize Memory Access**: Use CuTe copy operations
   ```cpp
   auto src = local_tile(global_tensor, thread_layout, thread_idx);
   auto dst = local_tile(shared_tensor, thread_layout, thread_idx);
   copy(src, dst);
   ```

3. **Pipeline Operations**: Overlap compute and memory
   ```cpp
   // Use CuTe's pipeline utilities for async operations
   copy_async(src, dst);
   cp_async_wait<0>();
   ```

### CUDA Baseline Optimization

1. **Shared Memory Tiling**: Reduce global memory accesses
2. **Warp-Level Primitives**: Use wmma for Tensor Cores
3. **Register Blocking**: Maximize data reuse in registers
4. **Memory Coalescing**: Ensure aligned, contiguous access patterns

### Profiling and Analysis

Use Nsight Compute to identify bottlenecks:

```bash
make tune-your_operator
```

Key metrics to monitor:
- **Compute Utilization**: `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- **Memory Bandwidth**: `dram__throughput.avg.pct_of_peak_sustained_elapsed`  
- **Tensor Core Usage**: `sm__inst_executed_pipe_tensor.sum`
- **Occupancy**: `sm__warps_active.avg.pct_of_peak_sustained_active`

## Best Practices

### Code Organization

1. **Single Responsibility**: One operator per file
2. **Clear Interfaces**: Simple, well-documented APIs
3. **Error Handling**: Check inputs and CUDA errors
4. **Template Design**: Support multiple data types
5. **Explicit Instantiation**: Reduce compile times

### Performance Guidelines

1. **Target High Utilization**: Aim for >80% compute/memory utilization
2. **Use Hardware Features**: Tensor Cores, async copy, shared memory
3. **Minimize Divergence**: Avoid branching within warps
4. **Optimize for Batch Size**: Larger batches often perform better
5. **Profile Early**: Use ncu to guide optimization

### Testing Strategy

1. **Correctness First**: Verify against reference implementation
2. **Edge Cases**: Test boundary conditions, error cases
3. **Performance Regression**: Ensure optimizations don't break correctness
4. **Multiple Sizes**: Test various problem sizes and batch sizes

### Documentation

1. **API Documentation**: Clear function signatures and parameters
2. **Algorithm Description**: Explain the computational approach
3. **Performance Characteristics**: Document expected performance
4. **Usage Examples**: Provide code examples

## Example: Complete Operator Implementation

For a complete example, see the existing `elementwise_add` operator:

- Header: `include/cutekernellib/operators/elementwise_add.h`
- Implementation: `src/operators/elementwise_add.cu`
- Baseline: `baselines/cuda/cuda_elementwise_add.cu`
- Benchmark: `benchmarks/bench_elementwise_add.cu`
- Test: `tests/test_elementwise_add.cpp`

This provides a working template you can adapt for your specific operator.

## Troubleshooting Tensor Core Implementations

### Common Tensor Core Issues

#### 1. Compilation Errors

**Error**: `error: no matching function for call to 'make_tiled_mma'`

**Solution**: Ensure correct MMA atom syntax and architecture support:
```cpp
#if __CUDA_ARCH__ >= 800
    using MMA_Atom = SM80_16x8x16_F32F16F16F32_TN;
#elif __CUDA_ARCH__ >= 750  
    using MMA_Atom = SM75_16x8x8_F32F16F16F32_TN;
#else
    #error "Tensor Cores require compute capability 7.5+"
#endif
```

**Error**: `error: identifier "cp_async_wait" is undefined`

**Solution**: Include proper CuTe headers and check architecture:
```cpp
#include <cute/arch/copy_sm80.hpp>  // For cp.async support
// cp.async requires SM80+ (Ampere)
```

#### 2. Runtime Issues

**Problem**: Kernel launches but produces incorrect results

**Debugging Steps**:
1. **Verify MMA atom dimensions match your problem size**:
   ```cpp
   // For SM80_16x8x16: A is 16x16, B is 16x8, C is 16x8
   static_assert(M % 16 == 0 && N % 8 == 0 && K % 16 == 0, 
                 "Dimensions must be compatible with MMA atom");
   ```

2. **Check thread block configuration**:
   ```cpp
   // Ensure thread block size matches TiledMMA requirements
   auto tiled_mma = make_tiled_mma(MMA_Atom{}, Layout<Shape<_8,_4,_1>>{});
   // This requires exactly 32 threads (8*4*1)
   ```

3. **Validate shared memory usage**:
   ```cpp
   // Check shared memory doesn't exceed limits (48KB on most GPUs)
   constexpr size_t smem_size = sizeof(smem_A) + sizeof(smem_B);
   static_assert(smem_size <= 48 * 1024, "Shared memory exceeds limit");
   ```

#### 3. Performance Issues

**Problem**: Low Tensor Core utilization in profiling

**Solutions**:
1. **Increase problem size**: Tensor Cores work best with large matrices
   ```cpp
   // Minimum recommended sizes for good utilization
   assert(M >= 128 && N >= 128 && K >= 128);
   ```

2. **Optimize tile sizes**: Match hardware capabilities
   ```cpp
   // Good tile sizes for Ampere
   constexpr int TILE_M = 128;  // Multiple of 16 (MMA atom M)
   constexpr int TILE_N = 128;  // Multiple of 8 (MMA atom N)  
   constexpr int TILE_K = 32;   // Multiple of 16 (MMA atom K)
   ```

3. **Check data types**: Mixed precision often performs better
   ```cpp
   // Use half for inputs, float for accumulation
   using InputType = __half;
   using AccumType = float;
   ```

### Debugging Tensor Core Kernels

#### 1. Enable Debug Information

```makefile
# Add debug flags to Makefile for debugging
DEBUG_FLAGS := -g -G -O0 -DDEBUG_TENSOR_CORES
```

#### 2. Add Debug Prints (Device Code)

```cpp
__global__ void debug_tensor_core_kernel(...) {
    // Print thread and block info
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Block (%d,%d), Thread (%d,%d)\n", 
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        printf("MMA atom size: %dx%dx%d\n", 
               size<0>(MMA_Atom{}), size<1>(MMA_Atom{}), size<2>(MMA_Atom{}));
    }
    
    // Print tensor shapes
    if (threadIdx.x == 0) {
        printf("Global A shape: (%d,%d)\n", size<0>(gA), size<1>(gA));
        printf("Global B shape: (%d,%d)\n", size<0>(gB), size<1>(gB));
        printf("Shared A shape: (%d,%d)\n", size<0>(sA), size<1>(sA));
    }
}
```

#### 3. Validate Intermediate Results

```cpp
// Add correctness checks within kernel
template<typename Fragment>
__device__ void validate_fragment(Fragment& frag, const char* name) {
    T sum = 0;
    for (int i = 0; i < size(frag); ++i) {
        sum += frag[i];
    }
    if (threadIdx.x == 0) {
        printf("%s fragment sum: %f\n", name, float(sum));
    }
}

// Use in kernel:
validate_fragment(tArA, "A");
validate_fragment(tBrB, "B");
validate_fragment(tCrC, "C");
```

### Profiling Tensor Core Performance

#### Key Metrics to Monitor

1. **Tensor Core Utilization**:
   ```bash
   # Check these metrics in ncu output:
   sm__inst_executed_pipe_tensor.sum                    # Total Tensor Core instructions
   sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active  # Utilization %
   ```

2. **Memory Efficiency**:
   ```bash
   dram__throughput.avg.pct_of_peak_sustained_elapsed   # Memory bandwidth %
   l1tex__throughput.avg.pct_of_peak_sustained_elapsed  # L1 cache efficiency
   ```

3. **Occupancy**:
   ```bash
   sm__warps_active.avg.pct_of_peak_sustained_active    # Warp occupancy
   sm__maximum_warps_per_active_cycle                   # Max warps per cycle
   ```

#### Optimization Targets

- **Tensor Core Utilization**: >90% for large matrices
- **Memory Bandwidth**: >80% of peak for memory-bound phases
- **Occupancy**: 50-100% depending on register/shared memory usage
- **Overall Performance**: ≥90% of cuBLAS for equivalent operations

## Troubleshooting

### Common Issues

1. **Compilation Errors**: Check CuTe header includes and CUDA arch flags
2. **Runtime Errors**: Verify memory allocation and kernel launch parameters
3. **Performance Issues**: Profile with ncu and optimize bottlenecks
4. **Correctness Issues**: Compare against reference implementation step-by-step
5. **Tensor Core Issues**: Verify MMA atom compatibility and data alignment

### Getting Help

1. **Check Existing Operators**: Use GEMM implementation as reference for Tensor Core patterns
2. **CuTe Documentation**: Refer to CUTLASS CuTe documentation and examples
3. **CUDA Best Practices**: Follow NVIDIA CUDA programming guidelines
4. **Profiling Tools**: Use Nsight Compute for performance analysis
5. **Tensor Core Guides**: Review NVIDIA's Tensor Core programming guides

---

This guide provides the foundation for extending CuTeKernelLib. Follow these patterns and you'll be able to add high-performance operators that integrate seamlessly with the existing framework.