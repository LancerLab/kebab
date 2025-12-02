/**
 * @file mbench_copy_gmem_to_smem.cu
 * @brief Microbenchmark for synchronous GMEM to SMEM copy implementations
 *
 * This benchmark compares different synchronous blocking load/store methods:
 * 1. Native CUDA C direct assignment
 * 2. Vectorized load (float4)
 * 3. Inline PTX (ld.global + st.shared)
 * 4. CuTe basic copy
 *
 * All variants are blocking (synchronous) transfers with __syncthreads().
 */

#include "microbench/microbench.h"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace kebab::microbench;
using namespace cute;

// ============================================================================
// Kernel Implementations
// ============================================================================

// Shared memory buffer size (48KB max per block typically)
constexpr int SMEM_SIZE = 32 * 1024;  // 32 KB shared memory buffer

/**
 * Variant 1: Native CUDA C - Direct assignment (scalar loads)
 */
__global__ void kernel_copy_native(const float* __restrict__ gmem,
                                    float* __restrict__ output,
                                    int n_elements) {
    __shared__ float smem[SMEM_SIZE / sizeof(float)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 4;  // Each thread handles 4 elements

    // Scalar load from GMEM to SMEM
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * blockDim.x;
        int gmem_idx = block_offset + idx;
        if (gmem_idx < n_elements && idx < (SMEM_SIZE / sizeof(float))) {
            smem[idx] = gmem[gmem_idx];
        }
    }
    __syncthreads();

    // Write back to output (to prevent compiler optimization)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * blockDim.x;
        int gmem_idx = block_offset + idx;
        if (gmem_idx < n_elements && idx < (SMEM_SIZE / sizeof(float))) {
            output[gmem_idx] = smem[idx];
        }
    }
}

/**
 * Variant 2: Vectorized load using float4 (128-bit loads)
 */
__global__ void kernel_copy_vectorized(const float* __restrict__ gmem,
                                        float* __restrict__ output,
                                        int n_elements) {
    __shared__ float smem[SMEM_SIZE / sizeof(float)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 4;

    const float4* gmem4 = reinterpret_cast<const float4*>(gmem + block_offset);
    float4* smem4 = reinterpret_cast<float4*>(smem);

    // Vectorized load: one float4 per thread
    if ((block_offset + tid * 4 + 3) < n_elements && (tid * 4 + 3) < (SMEM_SIZE / sizeof(float))) {
        smem4[tid] = gmem4[tid];
    }
    __syncthreads();

    // Write back using vectorized store
    float4* output4 = reinterpret_cast<float4*>(output + block_offset);
    if ((block_offset + tid * 4 + 3) < n_elements && (tid * 4 + 3) < (SMEM_SIZE / sizeof(float))) {
        output4[tid] = smem4[tid];
    }
}

/**
 * Variant 3: Inline PTX with cache hints (ld.global.cg + st.shared)
 */
__global__ void kernel_copy_ptx(const float* __restrict__ gmem,
                                 float* __restrict__ output,
                                 int n_elements) {
    __shared__ float smem[SMEM_SIZE / sizeof(float)];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x * 4;

    // PTX load with cache hints (to prevent compiler optimization)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * blockDim.x;
        int gmem_idx = block_offset + idx;
        if (gmem_idx < n_elements && idx < (SMEM_SIZE / sizeof(float))) {
            float val;
            const float* gptr = gmem + gmem_idx;
            float* sptr = smem + idx;
            // Use ld.global.cg for cache-global load, st.shared for shared store
            asm volatile(
                "ld.global.cg.f32 %0, [%1];\n\t"
                "st.shared.f32 [%2], %0;"
                : "=f"(val)
                : "l"(gptr), "r"((unsigned int)__cvta_generic_to_shared(sptr))
                : "memory"
            );
        }
    }
    __syncthreads();

    // Write back to output using inline PTX
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * blockDim.x;
        int gmem_idx = block_offset + idx;
        if (gmem_idx < n_elements && idx < (SMEM_SIZE / sizeof(float))) {
            float val;
            float* sptr = smem + idx;
            const float* gptr = output + gmem_idx;
            // Inline PTX: ld.shared from SMEM to register, st.global.cg to GMEM
            asm volatile(
                "ld.shared.f32 %0, [%1];\n\t"
                "st.global.cg.f32 [%2], %0;"
                : "=f"(val)  // Output: val from load (transient)
                : "r"((unsigned int)__cvta_generic_to_shared(sptr)), "l"(gptr)  // Shared: r (cast), Global: l
                : "memory"
            );
            (void)val;  // Suppress unused variable warning
        }
    }
}

/**
 * Variant 4: CuTe basic copy with Layout partitioning
 * Uses a thread layout to partition the data across threads
 */
__global__ void kernel_copy_cute(const float* __restrict__ gmem,
                                  float* __restrict__ output,
                                  int n_elements) {
    extern __shared__ float smem_cute[];

    int block_offset = blockIdx.x * blockDim.x * 4;
    int copy_size = min(blockDim.x * 4, n_elements - block_offset);
    if (copy_size <= 0) return;

    // Thread layout: 256 threads, each handling 4 elements
    // Layout: (256 threads) x (4 values per thread)
    auto thr_layout = make_layout(make_shape(Int<256>{}, Int<4>{}));

    // Create CuTe tensors with explicit layout
    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + block_offset),
                                    make_layout(make_shape(Int<256>{}, Int<4>{})));
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute),
                                    make_layout(make_shape(Int<256>{}, Int<4>{})));

    // Partition for this thread - use Layout not Shape
    auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    // Copy from GMEM to SMEM
    copy(thr_gmem, thr_smem);
    __syncthreads();

    // Write back
    auto out_tensor = make_tensor(make_gmem_ptr(output + block_offset),
                                   make_layout(make_shape(Int<256>{}, Int<4>{})));
    auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);
    copy(thr_smem, thr_out);
}

// ============================================================================
// Benchmark Runner
// ============================================================================

float getPeakMemoryBandwidth() {
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // Memory bandwidth = 2 * memory_clock_rate * memory_bus_width / 8 / 1e6 (GB/s)
    // Factor of 2 for DDR
    int memoryClockRate, memoryBusWidth;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    return 2.0f * memoryClockRate * (memoryBusWidth / 8.0f) / 1.0e6f;
}

void runBenchmark(size_t data_size_bytes, MicrobenchRunner& runner,
                  MicrobenchReport& report, float peak_bw) {
    size_t n_elements = data_size_bytes / sizeof(float);
    size_t n_bytes = n_elements * sizeof(float);

    // Allocate memory
    float* d_input;
    float* d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_input, n_bytes));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, n_bytes));

    // Initialize input
    std::vector<float> h_input(n_elements, 1.0f);
    MBENCH_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n_bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threads_per_block = 256;
    int elements_per_block = threads_per_block * 4;
    int num_blocks = (n_elements + elements_per_block - 1) / elements_per_block;

    // Shared memory size for CuTe kernel
    size_t smem_size = SMEM_SIZE;

    // Calculate effective bandwidth (read + write = 2x data size)
    size_t total_bytes = 2 * n_bytes;  // Read from GMEM + Write to GMEM

    // Variant 1: Native CUDA C
    {
        auto kernel = [=] { kernel_copy_native<<<num_blocks, threads_per_block>>>(d_input, d_output, n_elements); };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MicrobenchResult result;
        result.variant_name = "Native CUDA (scalar)";
        result.description = "Direct smem[i] = gmem[i] assignment";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        result.is_baseline = true;
        report.addResult(result);
    }

    // Variant 2: Vectorized (float4)
    {
        auto kernel = [=] { kernel_copy_vectorized<<<num_blocks, threads_per_block>>>(d_input, d_output, n_elements); };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MicrobenchResult result;
        result.variant_name = "Vectorized (float4)";
        result.description = "128-bit vectorized loads";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 3: Inline PTX
    {
        auto kernel = [=] { kernel_copy_ptx<<<num_blocks, threads_per_block>>>(d_input, d_output, n_elements); };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MicrobenchResult result;
        result.variant_name = "Inline PTX (cg)";
        result.description = "ld.global.cg + st.shared";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 4: CuTe basic copy
    {
        auto kernel = [=] { kernel_copy_cute<<<num_blocks, threads_per_block, smem_size>>>(d_input, d_output, n_elements); };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MicrobenchResult result;
        result.variant_name = "CuTe copy";
        result.description = "cute::copy with local_partition";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Cleanup
    MBENCH_CUDA_CHECK(cudaFree(d_input));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // Get peak memory bandwidth
    float peak_bw = getPeakMemoryBandwidth();

    MicrobenchRunner runner(20, 200);  // More iterations for stable results
    MicrobenchReport report("Synchronous GMEM→SMEM Copy", peak_bw);

    report.printHeader();
    std::cout << "Warmup iterations: " << runner.getWarmupIters() << "\n";
    std::cout << "Measurement iterations: " << runner.getMeasureIters() << "\n";

    // Test different data sizes
    std::vector<size_t> data_sizes = {
        64 * 1024,          // 64 KB
        256 * 1024,         // 256 KB
        1 * 1024 * 1024,    // 1 MB
        4 * 1024 * 1024,    // 4 MB
        16 * 1024 * 1024,   // 16 MB
        64 * 1024 * 1024    // 64 MB
    };

    for (size_t size : data_sizes) {
        runBenchmark(size, runner, report, peak_bw);
    }

    report.printTable();
    report.printSummary();

    return 0;
}
