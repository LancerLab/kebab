/**
 * @file copy_gmem_to_smem_2d_tiling.cu
 * @brief Microbenchmark for 2D tiling GMEM to SMEM copy implementations
 *
 * This benchmark compares different 2D tiling strategies for square matrix copies:
 * 1. Native CUDA C with 2D tiling (scalar loads)
 * 2. Vectorized float4 with 2D tiling
 * 3. Inline PTX with 2D tiling
 * 4. CuTe 2D tiled copy with UniversalCopy
 * 5. CuTe 2D tiled copy with float4 vectorization
 *
 * All variants process square matrices (NxN) with 2D thread blocks and 2D tiling.
 * Each tile is processed independently, enabling better cache locality and
 * reduced memory access conflicts compared to 1D linear access patterns.
 *
 * Key differences from 1D copy:
 * - 2D thread layout (e.g., 32x4 = 128 threads)
 * - 2D data layout (e.g., 256x256 = 65536 elements)
 * - Tile-based processing with row/column strides
 * - Better spatial locality for 2D data structures
 * - Reduced bank conflicts in shared memory
 */

#include "microbench/microbench.h"
#include "kebab/utils/data_size.h"
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>

using namespace kebab::microbench;
using namespace kebab::utils;
using namespace cute;

// ============================================================================
// 2D Tiling Configuration
// ============================================================================

// Tile dimensions (square tiles for simplicity)
constexpr int TILE_SIZE = 32;  // 32x32 tile
constexpr int TILE_ELEMENTS = TILE_SIZE * TILE_SIZE;  // 1024 elements per tile

// Thread block configuration
constexpr int BLOCK_DIM_X = 32;
constexpr int BLOCK_DIM_Y = 4;
// Total threads per block: 32 * 4 = 128

// Shared memory for one tile
constexpr int SMEM_SIZE_2D = TILE_ELEMENTS * sizeof(float);  // 4 KB per tile

/**
 * Variant 1: Native CUDA C with 2D tiling (scalar loads)
 *
 * Thread layout: 32x4 (128 threads)
 * Each thread loads multiple elements from a 32x32 tile
 * Coalesced access: threads in same warp load consecutive columns
 */
__global__ void kernel_copy_2d_native(const float* __restrict__ gmem,
                                      float* __restrict__ output,
                                      int matrix_size) {
    // give 32 as TILE_SIZE
    // sizeof(smem_data) = TILE_ELEMENTS * sizeof(float) = 1024 * 4 = 4096 bytes
    __shared__ float smem[TILE_ELEMENTS];

    // Thread indices (1D to 2D mapping)
    // for coalescing, x is major dimension, and we should try best to fulfill all elements
    // of major dimension in one transaction, we can thus calculate tx and ty as below
    int tid = threadIdx.x;
    int ty = tid / BLOCK_DIM_X;
    int tx = tid % BLOCK_DIM_X;

    // Calculate tile position
    // remember x should aligned with major dimension for coalescing
    int coord_x = blockIdx.x * TILE_SIZE;
    int coord_y = blockIdx.y * TILE_SIZE;

    // Load tile from GMEM to SMEM
    // Each thread loads TILE_SIZE/BLOCK_DIM_Y = 8 elements
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / BLOCK_DIM_Y; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = coord_y + smem_row;
        int smem_col = tx;  // 4 floats per thread
        int gmem_col = coord_x + tx;  // 4 floats per thread
        if (gmem_row < matrix_size && gmem_col < matrix_size) {
            smem[smem_row * TILE_SIZE + smem_col] = __ldg(&gmem[gmem_row * matrix_size + gmem_col]);
        }
    }
    __syncthreads();

    // Write tile back to GMEM
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / BLOCK_DIM_Y; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = coord_y + smem_row;
        int smem_col = tx;  // 4 floats per thread
        int gmem_col = coord_x + tx;  // 4 floats per thread
        if (gmem_row < matrix_size && gmem_col < matrix_size) {
            output[gmem_row * matrix_size + gmem_col] = smem[smem_row * TILE_SIZE + smem_col];
        }
    }
    __syncthreads();
}

/**
 * Variant 2: Vectorized float4 with 2D tiling
 *
 * Uses float4 for 128-bit loads/stores
 * Each thread loads 2 float4s (8 floats) per iteration
 */
__global__ void kernel_copy_2d_vectorized(const float* __restrict__ gmem,
                                          float* __restrict__ output,
                                          int matrix_size) {
    // Difference between v1 and v2/v4 memory transaction, unit from 1 elem to 4 elems
    // for same 2D memory shape.x should be divided by 4 logically; shape.y remains the same

    __shared__ float smem[TILE_ELEMENTS]; // 256
    float4* smem4 = reinterpret_cast<float4*>(smem);
    const float4* gmem4 = reinterpret_cast<const float4*>(gmem);
    float4* output4 = reinterpret_cast<float4*>(output);

    int matrix_size_x = matrix_size / 4; // 4
    int matrix_size_y = matrix_size; // 16
    constexpr int TILE_SIZE_X = TILE_SIZE / 4; // 4
    constexpr int TILE_SIZE_Y = TILE_SIZE; // 16
    constexpr int BLOCK_DIM_X_V4 = BLOCK_DIM_X / 4;  // 4
    constexpr int BLOCK_DIM_Y_V4 = BLOCK_DIM_Y * 4; // 8

    // Thread indices (1D to 2D mapping)
    // for coalescing, x is major dimension, and we should try best to fulfill all elements
    // of major dimension in one transaction, we can thus calculate tx and ty as below
    int tid = threadIdx.x;
    int ty_v4 = tid / BLOCK_DIM_X_V4; // 0 - 7
    int tx_v4 = tid % BLOCK_DIM_X_V4; // 0 - 3
    int ty = tid / BLOCK_DIM_X;
    int tx = tid % BLOCK_DIM_X;

    // Calculate tile position
    // remember x should aligned with major dimension for coalescing
    int coord_x_v4 = blockIdx.x * TILE_SIZE_X;
    int coord_y_v4 = blockIdx.y * TILE_SIZE_Y;
    int coord_x = blockIdx.x * TILE_SIZE;
    int coord_y = blockIdx.y * TILE_SIZE;

    // Load tile from GMEM to SMEM
    // Each thread loads TILE_SIZE/BLOCK_DIM_Y = 8 elements
    #pragma unroll
    for (int i = 0; i < TILE_SIZE_Y / BLOCK_DIM_Y_V4; ++i) {
        int smem_row = ty_v4 + i * BLOCK_DIM_Y_V4;
        int gmem_row = coord_y_v4 + smem_row;
        int smem_col = tx_v4;  // 4 floats per thread
        int gmem_col = coord_x_v4 + tx_v4;  // 4 floats per thread
        if (gmem_row < matrix_size_y && gmem_col < matrix_size_x) {
            smem4[smem_row * TILE_SIZE_X + smem_col] = __ldg(&gmem4[gmem_row * matrix_size_x + gmem_col]);
        }
    }
    __syncthreads();

    // Write tile back to GMEM
    #pragma unroll
    for (int i = 0; i < TILE_SIZE_Y / BLOCK_DIM_Y_V4; ++i) {
        int smem_row = ty_v4 + i * BLOCK_DIM_Y_V4;
        int gmem_row = coord_y_v4 + smem_row;
        int smem_col = tx_v4;  // 4 floats per thread
        int gmem_col = coord_x_v4 + tx_v4;  // 4 floats per thread
        if (gmem_row < matrix_size_y && gmem_col < matrix_size_x) {
            output4[gmem_row * matrix_size_x + gmem_col] = smem4[smem_row * TILE_SIZE_X + smem_col];
        }
    }
    __syncthreads();
}



/**
 * Variant 3: Inline PTX with 2D tiling
 *
 * Uses PTX ld.global.cg and st.shared for cache-friendly access
 */
__global__ void kernel_copy_2d_ptx(const float* __restrict__ gmem,
                                   float* __restrict__ output,
                                   int matrix_size) {
    extern __shared__ float smem_data[];
    float* smem = smem_data;

    int tile_x = blockIdx.x * TILE_SIZE;
    int tile_y = blockIdx.y * TILE_SIZE;

    int tid = threadIdx.x;
    int tx = tid % BLOCK_DIM_X;
    int ty = tid / BLOCK_DIM_X;

    // Load with PTX
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / BLOCK_DIM_Y; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int row = tile_y + smem_row;
        int col = tile_x + tx;

        if (row < matrix_size && col < matrix_size && smem_row < TILE_SIZE) {
            int gmem_idx = row * matrix_size + col;
            int smem_idx = smem_row * TILE_SIZE + tx;
            const float* gptr = gmem + gmem_idx;
            float* sptr = smem + smem_idx;

            asm volatile(
                "ld.global.cg.f32 %0, [%1];\n\t"
                "st.shared.f32 [%2], %0;"
                : "=f"(*(float*)sptr)
                : "l"(gptr), "r"((unsigned int)__cvta_generic_to_shared(sptr))
                : "memory"
            );
        }
    }
    __syncthreads();

    // Write back with PTX
    #pragma unroll
    for (int i = 0; i < TILE_SIZE / BLOCK_DIM_Y; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int row = tile_y + smem_row;
        int col = tile_x + tx;

        if (row < matrix_size && col < matrix_size && smem_row < TILE_SIZE) {
            int gmem_idx = row * matrix_size + col;
            int smem_idx = smem_row * TILE_SIZE + tx;
            float* sptr = smem + smem_idx;
            float* gptr = output + gmem_idx;

            asm volatile(
                "{\n\t"
                ".reg .f32 tmp;\n\t"
                "ld.shared.f32 tmp, [%0];\n\t"
                "st.global.cg.f32 [%1], tmp;\n\t"
                "}"
                :
                : "r"((unsigned int)__cvta_generic_to_shared(sptr)), "l"(gptr)
                : "memory"
            );
        }
    }
}

/**
 * Variant 4: CuTe 2D tiled copy with UniversalCopy
 *
 * Uses CuTe's make_tiled_copy with 2D layout
 * Thread layout: 32x4 (m-major stride)
 * Value layout: 1x8 (8 elements per thread)
 */
__global__ void kernel_copy_2d_cute_tiled(const float* __restrict__ gmem,
                                          float* __restrict__ output,
                                          int matrix_size) {
    extern __shared__ float smem_cute[];

    // define gmem layout and tensor, output has same layout with gmem
    auto gmem_layout = make_layout(make_shape(matrix_size, matrix_size));
    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem), gmem_layout);
    auto out_tensor = make_tensor(make_gmem_ptr(output), gmem_layout);

    // define smem layout and tensor
    auto smem_layout = make_layout(make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{}));
    auto smem_tensor = make_tensor(make_smem_ptr((float*)smem_cute), smem_layout);


    // Block level tiling on gmem
    // NOTE: local_tile = tiled_divide + indexing by make_coord
    // Tensor tiled_gmem_tensor = local_tile(gmem_tensor, make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{}), make_coord(blockIdx.x, blockIdx.y));
    Tensor tiled_gmem_tensors = tiled_divide(gmem_tensor, make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{})); // ((TILE_SIZE, TILE_SIZE), m, n)
    Tensor tiled_out_tensors = tiled_divide(out_tensor, make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{})); // ((TILE_SIZE, TILE_SIZE), m, n)
    Tensor tiled_gmem_tensor = tiled_gmem_tensors(make_coord(_, _), blockIdx.x, blockIdx.y); // (TILE_SIZE, TILE_SIZE)
    Tensor tiled_out_tensor = tiled_out_tensors(make_coord(_, _), blockIdx.x, blockIdx.y); // (TILE_SIZE, TILE_SIZE)

    // Create tiled copy
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<Int<32>, Int<4>>, Stride<Int<1>, Int<32>>>{},  // Thread layout, x=32, y=4
        Layout<Shape<Int<1>, Int<8>>>{}                              // Value layout
    );

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // partition_S, partition_D vs local_partition
    auto thr_gmem = thr_copy.partition_S(tiled_gmem_tensor);
    auto thr_smem = thr_copy.partition_D(smem_tensor);
    // auto thr_gmem = local_partition(tiled_gmem_tensor, make_shape(Int<32>{}, Int<4>{}), threadIdx.x);
    // auto thr_smem = local_partition(smem_tensor, make_shape(Int<32>{}, Int<4>{}), threadIdx.x);

    copy(tiled_copy, thr_gmem, thr_smem);
    __syncthreads();

    // Store tile
    auto thr_out = thr_copy.partition_D(tiled_out_tensor);

    copy(tiled_copy, thr_smem, thr_out);
    __syncthreads();
}

/**
 * Variant 5: CuTe 2D tiled copy with float4 vectorization
 *
 * Uses CuTe's make_tiled_copy with float for higher throughput
 * Thread layout: 32x4 = 128 threads (same as other variants)
 * Value layout: 1x8 (8 floats per thread)
 *
 * Note: Uses float (scalar) copy atom to avoid alignment issues with float4.
 * CuTe will automatically vectorize when possible.
 */
__global__ void kernel_copy_2d_cute_tiled_float4(const float* __restrict__ gmem,
                                                 float* __restrict__ output,
                                                 int matrix_size) {
    extern __shared__ float smem_cute[];

    // Define global memory layout and tensors
    auto gmem_layout = make_layout(make_shape(matrix_size, matrix_size));
    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem), gmem_layout);
    auto out_tensor = make_tensor(make_gmem_ptr(output), gmem_layout);

    // Define shared memory layout and tensor
    auto smem_layout = make_layout(make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{}));
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), smem_layout);

    // Block level tiling on gmem
    Tensor tiled_gmem_tensors = tiled_divide(gmem_tensor, make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{}));
    Tensor tiled_out_tensors = tiled_divide(out_tensor, make_shape(Int<TILE_SIZE>{}, Int<TILE_SIZE>{}));
    Tensor tiled_gmem_tensor = tiled_gmem_tensors(make_coord(_, _), blockIdx.x, blockIdx.y);
    Tensor tiled_out_tensor = tiled_out_tensors(make_coord(_, _), blockIdx.x, blockIdx.y);

    // Create tiled copy with float (scalar)
    // Thread layout: 32x4 = 128 threads (m-major: stride<1,32>)
    // Value layout: 1x8 = 8 floats per thread
    // Total per tile: 32*1 x 4*8 = 32 x 32 = 1024 floats = TILE_SIZE*TILE_SIZE
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<Int<32>, Int<4>>, Stride<Int<1>, Int<32>>>{},  // Thread layout: 32x4 m-major
        Layout<Shape<Int<1>, Int<8>>>{}                              // Value layout: 1x8 (8 floats)
    );

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Partition tensors for this thread
    auto thr_gmem = thr_copy.partition_S(tiled_gmem_tensor);
    auto thr_smem = thr_copy.partition_D(smem_tensor);

    // Copy GMEM → SMEM
    copy(tiled_copy, thr_gmem, thr_smem);
    __syncthreads();

    // Partition output tensor for this thread
    auto thr_out = thr_copy.partition_D(tiled_out_tensor);

    // Copy SMEM → GMEM
    copy(tiled_copy, thr_smem, thr_out);
    __syncthreads();
}

// ============================================================================
// Benchmark Runner
// ============================================================================

float getPeakMemoryBandwidth2D() {
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("sharedMemPerBlock: %zu bytes\n", prop.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin: %zu bytes\n", prop.sharedMemPerBlockOptin);
    int memoryClockRate, memoryBusWidth;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("memoryBusWidth: %zu bytes\n", memoryBusWidth);
    printf("memoryClockRate: %zu bytes\n", memoryClockRate);
    return 2.0f * memoryClockRate * (memoryBusWidth / 8.0f) / 1.0e6f;
}

bool verifyOutput2D(const float* h_input, const float* h_output, int matrix_size,
                    const std::string& variant_name) {
    // Sample verification: check corners and center
    std::vector<int> check_indices = {0, matrix_size - 1,
                                      matrix_size * (matrix_size / 2),
                                      matrix_size * matrix_size - 1};

    // for (int idx = 0; idx < matrix_size * matrix_size; idx++) {
    //     std::cout << "h_input[" << idx << "] = " << h_input[idx] << "; ";
    //     std::cout << "h_output[" << idx << "] = " << h_output[idx] << std::endl;
    // }
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

void runBenchmark2D(int matrix_size, MicrobenchRunner& runner,
                    MicrobenchReport& report, float peak_bw) {
    size_t n_elements = matrix_size * matrix_size;
    size_t n_bytes = n_elements * sizeof(float);

    // Allocate memory
    float* d_input;
    float* d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_input, n_bytes));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, n_bytes));

    // Initialize input
    std::vector<float> h_input(n_elements);
    std::vector<float> h_output(n_elements, 0.0f);
    for (size_t i = 0; i < n_elements; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    MBENCH_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n_bytes, cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    // each grid handles a tile: TILE_SIZE x TILE_SIZE, tail may remains
    int grid_x = (matrix_size + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (matrix_size + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_DIM_X * BLOCK_DIM_Y);  // 128 threads in 1D

    // Total bytes: read + write
    size_t total_bytes = 2 * n_bytes;

    std::cout << "\n  Matrix: " << matrix_size << "x" << matrix_size
              << " (" << formatBytes(n_bytes) << ")" << std::endl;
    std::cout << "  Grid: " << grid_x << "x" << grid_y << " | Tile: " << TILE_SIZE << "x" << TILE_SIZE << std::endl;
    std::cout << "  GMEM traffic: " << formatBytes(total_bytes) << " (read + write)" << std::endl;

    // Variant 1: Native 2D
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_native<<<grid, block, SMEM_SIZE_2D>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput2D(h_input.data(), h_output.data(), matrix_size, "Native 2D");

        MicrobenchResult result;
        result.variant_name = "Native 2D (scalar)";
        result.description = correct ? "2D tiling, 32x32 tiles, scalar loads" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        result.is_baseline = true;
        report.addResult(result);
    }

    // Variant 2: Vectorized 2D (float4, 128-bit loads)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_vectorized<<<grid, block, SMEM_SIZE_2D>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput2D(h_input.data(), h_output.data(), matrix_size, "Vectorized 2D");

        MicrobenchResult result;
        result.variant_name = "Vectorized 2D (float4)";
        result.description = correct ? "2D tiling, 32x32 tiles, float4 loads" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 3: PTX 2D (ld.global.cg with cache hints)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_ptx<<<grid, block, SMEM_SIZE_2D>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput2D(h_input.data(), h_output.data(), matrix_size, "PTX 2D");

        MicrobenchResult result;
        result.variant_name = "PTX 2D (ld.cg)";
        result.description = correct ? "2D tiling, 32x32 tiles, PTX ld.global.cg" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 4: CuTe 2D tiled copy with UniversalCopy
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_cute_tiled<<<grid, block, SMEM_SIZE_2D>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput2D(h_input.data(), h_output.data(), matrix_size, "CuTe 2D");

        MicrobenchResult result;
        result.variant_name = "CuTe 2D tiled";
        result.description = correct ? "make_tiled_copy, UniversalCopy, 2D layout" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 5: CuTe 2D tiled copy with float4 vectorization
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_cute_tiled_float4<<<grid, block, SMEM_SIZE_2D>>>(d_input, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput2D(h_input.data(), h_output.data(), matrix_size, "CuTe 2D float4");

        MicrobenchResult result;
        result.variant_name = "CuTe 2D tiled (float4)";
        result.description = correct ? "make_tiled_copy, UniversalCopy, 2D layout, 32 floats/thread" : "VERIFICATION FAILED";
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
    // Get device info
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Get peak memory bandwidth
    float peak_bw = getPeakMemoryBandwidth2D();

    MicrobenchRunner runner(20, 200);  // 20 warmup, 200 measurement iterations
    MicrobenchReport report("2D Tiling GMEM→SMEM→GMEM Copy", peak_bw);

    report.printHeader();
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Warmup: 20 iterations | Measurement: 200 iterations\n\n";

    // Test different matrix sizes
    std::vector<int> matrix_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    // std::vector<int> matrix_sizes = {32};

    for (int size : matrix_sizes) {
        std::cout << "=== Matrix Size: " << size << "x" << size << " ===\n";
        runBenchmark2D(size, runner, report, peak_bw);
    }

    report.printTable();
    report.printSummary();
    return 0;
}
