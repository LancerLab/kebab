/**
 * @file copy_gmem_to_smem_2d_cp_async.cu
 * @brief Microbenchmark for 2D tiling GMEM to SMEM copy using TRUE cp.async
 *
 * This benchmark demonstrates the REAL cp.async hardware feature:
 * - Uses cp.async.ca.shared.global PTX instruction (bypasses registers)
 * - Uses commit_group/wait_group for async synchronization
 * - Demonstrates latency hiding capability
 *
 * cp.async key features:
 * 1. Direct GMEM to SMEM transfer (no intermediate registers)
 * 2. Asynchronous execution (returns immediately)
 * 3. Commit/Wait mechanism for synchronization
 * 4. Available on SM_80+ (Ampere and later)
 *
 * Variants:
 * 1. cp.async with CUDA pipeline intrinsics (__pipeline_memcpy_async)
 * 2. cp.async with PTX inline assembly (cp.async.ca.shared.global)
 */

#include "microbench/microbench.h"
#include "kebab/utils/data_size.h"
#include <cuda_runtime.h>

using namespace kebab::microbench;
using namespace kebab::utils;

// ============================================================================
// 2D Tiling Configuration
// ============================================================================

// cp.async on SM_90 only supports 16-byte transfers, so we use float4
constexpr int TILE_SIZE = 32;
constexpr int TILE_ELEMENTS = TILE_SIZE * TILE_SIZE;
constexpr int BLOCK_DIM_X = 8;   // 8 threads per row (each handles float4 = 4 floats)
constexpr int BLOCK_DIM_Y = 16;  // 16 rows per iteration
constexpr int THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y;  // 128 threads
constexpr int ROWS_PER_THREAD = TILE_SIZE / BLOCK_DIM_Y;      // 2 rows per thread

/**
 * Variant 1: cp.async with cp.async.cg (cache global) PTX
 *
 * Uses cp.async.cg.shared.global for async GMEM to SMEM copy
 * - cg: cache global (bypass L1, use L2 only)
 * - Uses 16-byte (float4) transfers as required by SM_90
 * - Suitable for streaming data that won't be reused soon
 */
__global__ void kernel_copy_2d_cp_async_cg(const float* __restrict__ gmem,
                                            float* __restrict__ output,
                                            int matrix_size) {
    __shared__ __align__(16) float smem[TILE_ELEMENTS];

    int tid = threadIdx.x;
    int tx = tid % BLOCK_DIM_X;  // 0-7 (each handles 4 floats)
    int ty = tid / BLOCK_DIM_X;  // 0-15

    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;

    // Phase 1: Async copy GMEM to SMEM using cp.async.cg (16-byte transfers)
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = tile_y + smem_row;
        int gmem_col = tile_x + tx * 4;  // Each thread handles 4 floats

        if (gmem_row < matrix_size && gmem_col + 3 < matrix_size) {
            float* smem_ptr = &smem[smem_row * TILE_SIZE + tx * 4];
            const float* gmem_ptr = &gmem[gmem_row * matrix_size + gmem_col];

            // cp.async.cg: cache global (bypass L1), 16 bytes = float4
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;"
                :
                : "r"((unsigned int)__cvta_generic_to_shared(smem_ptr)),
                  "l"(gmem_ptr)
                : "memory"
            );
        }
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // Phase 2: Write back to GMEM (vectorized)
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = tile_y + smem_row;
        int gmem_col = tile_x + tx * 4;

        if (gmem_row < matrix_size && gmem_col + 3 < matrix_size) {
            float4* smem_ptr = reinterpret_cast<float4*>(&smem[smem_row * TILE_SIZE + tx * 4]);
            float4* out_ptr = reinterpret_cast<float4*>(&output[gmem_row * matrix_size + gmem_col]);
            *out_ptr = *smem_ptr;
        }
    }
}

/**
 * Variant 2: cp.async with cp.async.ca (cache all) PTX
 *
 * Uses raw PTX instructions:
 * - cp.async.ca.shared.global: Async copy with L1 cache hint
 * - Uses 16-byte (float4) transfers as required by SM_90
 * - cp.async.commit_group: Commit current group
 * - cp.async.wait_group 0: Wait for all groups
 */
__global__ void kernel_copy_2d_cp_async_ca(const float* __restrict__ gmem,
                                            float* __restrict__ output,
                                            int matrix_size) {
    __shared__ __align__(16) float smem[TILE_ELEMENTS];

    int tid = threadIdx.x;
    int tx = tid % BLOCK_DIM_X;  // 0-7 (each handles 4 floats)
    int ty = tid / BLOCK_DIM_X;  // 0-15

    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;

    // Phase 1: Async copy GMEM to SMEM using cp.async.ca (16-byte transfers)
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = tile_y + smem_row;
        int gmem_col = tile_x + tx * 4;  // Each thread handles 4 floats

        if (gmem_row < matrix_size && gmem_col + 3 < matrix_size) {
            float* smem_ptr = &smem[smem_row * TILE_SIZE + tx * 4];
            const float* gmem_ptr = &gmem[gmem_row * matrix_size + gmem_col];

            // cp.async.ca: cache all (use L1), 16 bytes = float4
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;"
                :
                : "r"((unsigned int)__cvta_generic_to_shared(smem_ptr)),
                  "l"(gmem_ptr)
                : "memory"
            );
        }
    }

    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // Phase 2: Write back to GMEM (vectorized)
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = tile_y + smem_row;
        int gmem_col = tile_x + tx * 4;

        if (gmem_row < matrix_size && gmem_col + 3 < matrix_size) {
            float4* smem_ptr = reinterpret_cast<float4*>(&smem[smem_row * TILE_SIZE + tx * 4]);
            float4* out_ptr = reinterpret_cast<float4*>(&output[gmem_row * matrix_size + gmem_col]);
            *out_ptr = *smem_ptr;
        }
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

float getPeakMemoryBandwidthCpAsync() {
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memoryClockRate, memoryBusWidth;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    return 2.0f * memoryClockRate * (memoryBusWidth / 8.0f) / 1.0e6f;
}

bool verifyOutputCpAsync(const float* h_input, const float* h_output, int matrix_size,
                         const std::string& variant_name) {
    std::vector<int> check_indices = {0, matrix_size - 1,
                                      matrix_size * (matrix_size / 2),
                                      matrix_size * matrix_size - 1};

    for (int idx : check_indices) {
        if (idx >= matrix_size * matrix_size) continue;
        if (h_input[idx] != h_output[idx]) {
            std::cerr << "  ERROR: " << variant_name << " verification failed at index "
                      << idx << ": expected " << h_input[idx] << ", got " << h_output[idx] << std::endl;
            return false;
        }
    }
    return true;
}

void runBenchmarkCpAsync(int matrix_size, MicrobenchRunner& runner,
                         MicrobenchReport& report, float peak_bw) {
    size_t n_elements = matrix_size * matrix_size;
    size_t n_bytes = n_elements * sizeof(float);

    float* d_input;
    float* d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_input, n_bytes));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, n_bytes));

    std::vector<float> h_input(n_elements);
    std::vector<float> h_output(n_elements, 0.0f);
    for (size_t i = 0; i < n_elements; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    MBENCH_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n_bytes, cudaMemcpyHostToDevice));

    int grid_x = (matrix_size + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (matrix_size + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_DIM_X * BLOCK_DIM_Y);

    size_t total_bytes = 2 * n_bytes;

    std::cout << "\n  Matrix: " << matrix_size << "x" << matrix_size
              << " (" << formatBytes(n_bytes) << ")" << std::endl;
    std::cout << "  Grid: " << grid_x << "x" << grid_y << " | Tile: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
    std::cout << "  GMEM traffic: " << formatBytes(total_bytes) << " (read + write)" << std::endl;

    // Variant 1: cp.async.cg (cache global - bypass L1)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_cp_async_cg<<<grid, block>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutputCpAsync(h_input.data(), h_output.data(), matrix_size, "cp.async.cg");

        MicrobenchResult result;
        result.variant_name = "cp.async.cg (PTX)";
        result.description = correct ? "TRUE cp.async with cp.async.cg.shared.global (bypass L1)" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        result.is_baseline = true;
        report.addResult(result);
    }

    // Variant 2: cp.async.ca (cache all - use L1)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_cp_async_ca<<<grid, block>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutputCpAsync(h_input.data(), h_output.data(), matrix_size, "cp.async.ca");

        MicrobenchResult result;
        result.variant_name = "cp.async.ca (PTX)";
        result.description = correct ? "TRUE cp.async with cp.async.ca.shared.global (use L1)" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    MBENCH_CUDA_CHECK(cudaFree(d_input));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
}

int main() {
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    float peak_bw = getPeakMemoryBandwidthCpAsync();

    MicrobenchRunner runner(20, 200);
    MicrobenchReport report("2D Tiling GMEM→SMEM→GMEM Copy (TRUE cp.async)", peak_bw);

    report.printHeader();
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Peak Memory BW: " << peak_bw << " GB/s\n";
    std::cout << "Warmup: 20 iterations | Measurement: 200 iterations\n";
    std::cout << "\n*** Using TRUE cp.async (bypasses registers, direct GMEM→SMEM) ***\n\n";

    std::vector<int> matrix_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int size : matrix_sizes) {
        std::cout << "=== Matrix Size: " << size << "x" << size << " ===\n";
        runBenchmarkCpAsync(size, runner, report, peak_bw);
    }

    report.printTable();
    report.printSummary();
    return 0;
}
