/**
 * @file copy_gmem_to_smem_2d_tma_ptx.cu
 * @brief Microbenchmark for 2D tiling GMEM to SMEM copy using TRUE TMA with PTX
 *
 * This benchmark demonstrates the REAL TMA (Tensor Memory Accelerator) hardware feature:
 * - Uses cp.async.bulk.tensor PTX instruction
 * - Uses CUtensorMap descriptor created on host
 * - Uses mbarrier for synchronization
 * - Dedicated hardware unit (doesn't use SM compute resources)
 *
 * TMA key features:
 * 1. Dedicated hardware unit for tensor transfers
 * 2. Descriptor-based: CUtensorMap describes tensor properties
 * 3. Multi-dimensional native support (1D-5D)
 * 4. Bulk transfer: entire tile in one instruction
 * 5. mbarrier synchronization (different from cp.async commit/wait)
 * 6. Available on SM_90+ (Hopper and later)
 *
 * Variants:
 * 1. TMA Load with PTX (cp.async.bulk.tensor.2d)
 * 2. TMA Load with vectorized writeback
 */

#include "microbench/microbench.h"
#include "kebab/utils/data_size.h"
#include <cuda_runtime.h>
#include <cuda.h>

using namespace kebab::microbench;
using namespace kebab::utils;

// ============================================================================
// 2D Tiling Configuration
// ============================================================================

constexpr int TILE_M = 32;  // Tile rows
constexpr int TILE_N = 32;  // Tile columns
constexpr int TILE_ELEMENTS = TILE_M * TILE_N;
constexpr int BLOCK_DIM_X = 8;   // 8 threads per row (each handles float4)
constexpr int BLOCK_DIM_Y = 16;  // 16 rows per iteration
constexpr int THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y;  // 128 threads
constexpr int ROWS_PER_THREAD = TILE_M / BLOCK_DIM_Y;         // 2 rows per thread
constexpr int TILE_BYTES = TILE_ELEMENTS * sizeof(float);

// Shared memory structure with mbarrier
struct SharedStorage {
    __align__(128) float data[TILE_ELEMENTS];
    __align__(8) uint64_t mbarrier;
};

/**
 * Initialize mbarrier for TMA transaction
 */
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int arrive_count) {
    asm volatile(
        "mbarrier.init.shared.b64 [%0], %1;"
        :
        : "r"((unsigned int)__cvta_generic_to_shared(mbar)), "r"(arrive_count)
        : "memory"
    );
}

/**
 * Arrive at mbarrier and expect transaction bytes
 */
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, int tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :
        : "r"((unsigned int)__cvta_generic_to_shared(mbar)), "r"(tx_bytes)
        : "memory"
    );
}

/**
 * Wait on mbarrier with phase
 */
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, int phase) {
    // Spin-wait on mbarrier
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT_LOOP:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_LOOP;\n\t"
        "}"
        :
        : "r"((unsigned int)__cvta_generic_to_shared(mbar)), "r"(phase)
        : "memory"
    );
}

/**
 * TMA 2D Load using PTX cp.async.bulk.tensor instruction
 *
 * @param desc_ptr  Pointer to CUtensorMap descriptor
 * @param mbar      Pointer to mbarrier in shared memory
 * @param smem_ptr  Destination in shared memory
 * @param coord_x   X coordinate in tensor
 * @param coord_y   Y coordinate in tensor
 */
__device__ __forceinline__ void tma_load_2d(
    const void* desc_ptr,
    uint64_t* mbar,
    void* smem_ptr,
    int coord_x,
    int coord_y
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :
        : "r"((unsigned int)__cvta_generic_to_shared(smem_ptr)),
          "l"(desc_ptr),
          "r"((unsigned int)__cvta_generic_to_shared(mbar)),
          "r"(coord_x), "r"(coord_y)
        : "memory"
    );
}

/**
 * Variant 1: TMA Load with PTX
 *
 * Uses TRUE TMA hardware:
 * - cp.async.bulk.tensor.2d for bulk tensor transfer
 * - mbarrier for synchronization
 * - Only thread 0 issues the TMA load (hardware handles distribution)
 */
__global__ void kernel_copy_2d_tma_ptx(
    const CUtensorMap* __restrict__ tensorMap,
    float* __restrict__ output,
    int matrix_size
) {
    extern __shared__ SharedStorage smem_storage[];
    SharedStorage& smem = smem_storage[0];

    int tid = threadIdx.x;
    int tx = tid % BLOCK_DIM_X;
    int ty = tid / BLOCK_DIM_X;

    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    // Phase 1: Initialize mbarrier and issue TMA load
    if (tid == 0) {
        // Initialize mbarrier with 1 arrival (this thread)
        mbarrier_init(&smem.mbarrier, 1);

        // Expect TILE_BYTES of transaction data
        mbarrier_arrive_expect_tx(&smem.mbarrier, TILE_BYTES);

        // Issue TMA 2D load - entire tile in one instruction!
        tma_load_2d(tensorMap, &smem.mbarrier, smem.data, tile_x, tile_y);
    }
    __syncthreads();

    // Wait for TMA to complete
    if (tid == 0) {
        mbarrier_wait(&smem.mbarrier, 0);
    }
    __syncthreads();

    // Phase 2: Write back to GMEM (vectorized)
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = tile_y + smem_row;
        int gmem_col = tile_x + tx * 4;

        if (gmem_row < matrix_size && gmem_col + 3 < matrix_size) {
            float4* smem_ptr = reinterpret_cast<float4*>(&smem.data[smem_row * TILE_N + tx * 4]);
            float4* out_ptr = reinterpret_cast<float4*>(&output[gmem_row * matrix_size + gmem_col]);
            *out_ptr = *smem_ptr;
        }
    }
}

/**
 * Variant 2: TMA Load with vectorized writeback (same as variant 1 but explicit)
 */
__global__ void kernel_copy_2d_tma_ptx_vec(
    const CUtensorMap* __restrict__ tensorMap,
    float* __restrict__ output,
    int matrix_size
) {
    extern __shared__ SharedStorage smem_storage[];
    SharedStorage& smem = smem_storage[0];

    int tid = threadIdx.x;
    int tx = tid % BLOCK_DIM_X;
    int ty = tid / BLOCK_DIM_X;

    int tile_x = blockIdx.x * TILE_N;
    int tile_y = blockIdx.y * TILE_M;

    // Phase 1: Initialize mbarrier and issue TMA load
    if (tid == 0) {
        mbarrier_init(&smem.mbarrier, 1);
        mbarrier_arrive_expect_tx(&smem.mbarrier, TILE_BYTES);
        tma_load_2d(tensorMap, &smem.mbarrier, smem.data, tile_x, tile_y);
    }
    __syncthreads();

    if (tid == 0) {
        mbarrier_wait(&smem.mbarrier, 0);
    }
    __syncthreads();

    // Phase 2: Write back to GMEM using float4 stores
    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int smem_row = ty + i * BLOCK_DIM_Y;
        int gmem_row = tile_y + smem_row;
        int gmem_col = tile_x + tx * 4;

        if (gmem_row < matrix_size && gmem_col + 3 < matrix_size) {
            float4 val = *reinterpret_cast<float4*>(&smem.data[smem_row * TILE_N + tx * 4]);

            // Use PTX for vectorized store
            asm volatile(
                "st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                :
                : "l"(&output[gmem_row * matrix_size + gmem_col]),
                  "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
                : "memory"
            );
        }
    }
}

// ============================================================================
// TMA Descriptor Creation (Host-side)
// ============================================================================

/**
 * Create TMA descriptor for 2D tensor
 *
 * TMA requires a CUtensorMap descriptor that describes:
 * - Tensor dimensions and strides
 * - Element type and size
 * - Tile (box) dimensions for transfer
 * - Swizzle mode for shared memory
 */
CUtensorMap createTmaDescriptor2D(
    const float* gmem_ptr,
    int tensor_rows,
    int tensor_cols
) {
    CUtensorMap tensorMap;

    // Tensor dimensions (in elements)
    uint64_t globalDim[2] = {
        static_cast<uint64_t>(tensor_cols),  // dim 0: columns (contiguous)
        static_cast<uint64_t>(tensor_rows)   // dim 1: rows
    };

    // Tensor strides (in bytes) - stride[0] is implicit (element size)
    uint64_t globalStride[1] = {
        static_cast<uint64_t>(tensor_cols * sizeof(float))  // stride between rows
    };

    // Box (tile) dimensions for TMA transfer
    uint32_t boxDim[2] = {
        static_cast<uint32_t>(TILE_N),  // box dim 0: columns
        static_cast<uint32_t>(TILE_M)   // box dim 1: rows
    };

    // Element strides within box (usually 1)
    uint32_t elementStride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensorMap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,     // Element type
        2,                                    // Tensor rank (2D)
        const_cast<void*>(static_cast<const void*>(gmem_ptr)),  // Global memory address
        globalDim,                            // Tensor dimensions
        globalStride,                         // Tensor strides (bytes)
        boxDim,                               // Box dimensions
        elementStride,                        // Element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,       // No interleaving
        CU_TENSOR_MAP_SWIZZLE_NONE,          // No swizzle
        CU_TENSOR_MAP_L2_PROMOTION_NONE,     // No L2 promotion
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE    // No OOB fill
    );

    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        std::cerr << "cuTensorMapEncodeTiled failed: " << errStr << std::endl;
    }

    return tensorMap;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

float getPeakMemoryBandwidthTmaPtx() {
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int memoryClockRate, memoryBusWidth;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    return 2.0f * memoryClockRate * (memoryBusWidth / 8.0f) / 1.0e6f;
}

bool verifyOutputTmaPtx(const float* h_input, const float* h_output, int matrix_size,
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

void runBenchmarkTmaPtx(int matrix_size, MicrobenchRunner& runner,
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

    // Create TMA descriptor on host
    CUtensorMap tensorMap = createTmaDescriptor2D(d_input, matrix_size, matrix_size);

    // Copy TMA descriptor to device
    CUtensorMap* d_tensorMap;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_tensorMap, sizeof(CUtensorMap)));
    MBENCH_CUDA_CHECK(cudaMemcpy(d_tensorMap, &tensorMap, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

    int grid_x = (matrix_size + TILE_N - 1) / TILE_N;
    int grid_y = (matrix_size + TILE_M - 1) / TILE_M;
    dim3 grid(grid_x, grid_y);
    dim3 block(THREADS_PER_BLOCK);

    size_t smem_size = sizeof(SharedStorage);
    size_t total_bytes = 2 * n_bytes;

    std::cout << "\n  Matrix: " << matrix_size << "x" << matrix_size
              << " (" << formatBytes(n_bytes) << ")" << std::endl;
    std::cout << "  Grid: " << grid_x << "x" << grid_y << " | Tile: " << TILE_M << "x" << TILE_N << std::endl;
    std::cout << "  GMEM traffic: " << formatBytes(total_bytes) << " (read + write)" << std::endl;
    std::cout << "  *** Using TRUE TMA (cp.async.bulk.tensor.2d) ***" << std::endl;

    // Variant 1: TMA with PTX
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_tma_ptx<<<grid, block, smem_size>>>(d_tensorMap, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutputTmaPtx(h_input.data(), h_output.data(), matrix_size, "TMA PTX");

        MicrobenchResult result;
        result.variant_name = "TMA (PTX)";
        result.description = correct ? "TRUE TMA with cp.async.bulk.tensor.2d" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        result.is_baseline = true;
        report.addResult(result);
    }

    // Variant 2: TMA with PTX (vectorized writeback)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));
        auto kernel = [=] {
            kernel_copy_2d_tma_ptx_vec<<<grid, block, smem_size>>>(d_tensorMap, d_output, matrix_size);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutputTmaPtx(h_input.data(), h_output.data(), matrix_size, "TMA PTX Vec");

        MicrobenchResult result;
        result.variant_name = "TMA (PTX, vectorized)";
        result.description = correct ? "TRUE TMA with vectorized st.global.v4" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    MBENCH_CUDA_CHECK(cudaFree(d_tensorMap));
    MBENCH_CUDA_CHECK(cudaFree(d_input));
    MBENCH_CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // Initialize CUDA driver API
    CUresult cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "cuInit failed" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Check for SM_90+ (Hopper)
    if (prop.major < 9) {
        std::cerr << "TMA requires SM_90+ (Hopper). Current device: SM_"
                  << prop.major << prop.minor << std::endl;
        return 1;
    }

    float peak_bw = getPeakMemoryBandwidthTmaPtx();

    MicrobenchRunner runner(20, 200);
    MicrobenchReport report("2D Tiling GMEM→SMEM→GMEM Copy (TRUE TMA with PTX)", peak_bw);

    report.printHeader();
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Peak Memory BW: " << peak_bw << " GB/s\n";
    std::cout << "Warmup: 20 iterations | Measurement: 200 iterations\n\n";
    std::cout << "*** Using TRUE TMA (Tensor Memory Accelerator) ***\n";
    std::cout << "*** TMA uses dedicated hardware, bypasses SM compute ***\n\n";

    std::vector<int> matrix_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int size : matrix_sizes) {
        std::cout << "=== Matrix Size: " << size << "x" << size << " ===\n";
        runBenchmarkTmaPtx(size, runner, report, peak_bw);
    }

    report.printTable();
    report.printSummary();
    return 0;
}
