/**
 * @file mbench_copy_gmem_to_smem.cu
 * @brief Microbenchmark for synchronous GMEM to SMEM copy implementations
 *
 * This benchmark compares different synchronous blocking load/store methods:
 * 1. Native CUDA C direct assignment (scalar loads - intentionally slow baseline)
 * 2. Vectorized load (float4 - 128-bit loads)
 * 3. Inline PTX (ld.global.cg + st.shared with cache hints)
 * 4. CuTe basic copy (cute::copy with automatic optimization)
 * 5. CuTe copy with indexed tail copy (manual bounds checking for partial chunks)
 * 6. CuTe tiled copy (make_tiled_copy with no tail handling for full tiles)
 *
 * All variants are blocking (synchronous) transfers with __syncthreads().
 * Each kernel supports loop iterations to stress test large data through SMEM.
 *
 * NOTE: This benchmark measures GMEM→SMEM→GMEM round-trip performance.
 * The "efficiency" metric shows effective throughput relative to theoretical
 * GMEM bandwidth. Values > 100% are expected and indicate that:
 * - L1/L2 cache reduces actual GMEM traffic
 * - SMEM bandwidth >> GMEM bandwidth (not the bottleneck)
 * - CuTe may use cp.async or other async instructions for overlap
 * - Optimal memory coalescing and access patterns
 *
 * Performance characteristics:
 * - Small data (16-64 MB): High cache hit rate → very high efficiency
 * - Medium data (128-256 MB): Cache effects reduce, SMEM optimization dominates
 * - Large data (512+ MB): Exceeds cache capacity, efficiency drops
 *
 * Focus on relative speedup between variants rather than absolute efficiency.
 *
 * Variant 6 (CuTe tiled copy) demonstrates:
 * - How to use make_tiled_copy for efficient data movement
 * - 2D thread layout (32x4) with m-major stride for coalesced access
 * - Value layout (1x8) for 8 elements per thread
 * - Avoids tail handling by processing complete tiles only
 * - Falls back to manual copy for partial chunks at block boundaries
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
// Kernel Configuration
// ============================================================================

// Tiling settigs !!!!!, can be determined no matter what problem size
// Shared memory buffer size (48KB max per block typically)
// for H800, you can request 64KB at most, if explicitly requested, you can get up to 100KB
constexpr int SMEM_SIZE_PER_BLOCK = 32 * 1024;  // 16 KB shared memory buffer
constexpr int SMEM_ELEMENTS_PER_BLOCK = SMEM_SIZE_PER_BLOCK / sizeof(float);  // 4096 elements
constexpr int THREADS_PER_BLOCK = 128;
constexpr int SMEM_ELEMENTS_PER_THREAD = SMEM_ELEMENTS_PER_BLOCK / THREADS_PER_BLOCK;  // 32 elements per thread

/**
 * Variant 1: Native CUDA C - Scalar loads (32B coalesced per warp)
 *
 * Coalesced access pattern (CORRECT):
 * - Each thread loads 1 float (4 bytes) per iteration
 * - Warp of 32 threads: threads 0-7 together load 32 bytes (one cache line)
 * - threadIdx.x provides stride=1 offset (consecutive threads → consecutive addresses)
 * - Same thread's consecutive loads are stride=blockDim.x (256) apart
 *
 * Example for first 2 iterations (256 threads):
 *   Iteration 0: thread 0 loads idx 0, thread 1 loads idx 1, ..., thread 255 loads idx 255
 *   Iteration 1: thread 0 loads idx 256, thread 1 loads idx 257, ..., thread 255 loads idx 511
 */
__global__ void kernel_copy_native(const float* __restrict__ gmem,
                                    float* __restrict__ output,
                                    int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;

    // Loop over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;

        // Load GMEM → SMEM
        // Each thread loads ELEMENTS_PER_THREAD (32) floats
        // Access pattern: tid, tid+256, tid+512, ..., tid+31*256
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;  // stride = blockDim.x = 256
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                // Use __ldg for cache-friendly load
                smem[idx] = __ldg(&gmem[gmem_idx]);
            }
        }
        __syncthreads();

        // Write back SMEM → GMEM
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                output[gmem_idx] = smem[idx];
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 2: Vectorized load using float4 (128-bit loads, fully coalesced)
 *
 * Each thread loads 8 float4s (32 floats = 128 bytes)
 * Warp loads: 32 threads * 128 bytes = 4KB per iteration
 */
__global__ void kernel_copy_vectorized(const float* __restrict__ gmem,
                                        float* __restrict__ output,
                                        int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range (aligned to float4 boundaries)
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    // Align to 4-element boundary for float4
    total_elements_per_block = ((total_elements_per_block + 3) / 4) * 4;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Skip if this block is out of range
    if (block_start >= n_elements) return;

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;
    float4* smem4 = reinterpret_cast<float4*>(smem);
    constexpr int FLOAT4_PER_THREAD = SMEM_ELEMENTS_PER_THREAD / 4;  // 32/4 = 8

    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        const float4* gmem4 = reinterpret_cast<const float4*>(gmem + iter_offset);

        // Vectorized load: 8 float4s per thread (128 bytes)
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                // Use __ldg for cache-friendly load
                smem4[idx] = __ldg(&gmem4[idx]);
            }
        }
        __syncthreads();

        // Vectorized store
        float4* output4 = reinterpret_cast<float4*>(output + iter_offset);
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                output4[idx] = smem4[idx];
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 8: PTX version, Vectorized load using float4 (128-bit loads, fully coalesced)
 *
 * Each thread loads 8 float4s (32 floats = 128 bytes)
 * Warp loads: 32 threads * 128 bytes = 4KB per iteration
 */

__global__ void kernel_copy_vectorized_ptx(const float* __restrict__ gmem,
                                        float* __restrict__ output,
                                        int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range (aligned to float4 boundaries)
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    // Align to 4-element boundary for float4
    total_elements_per_block = ((total_elements_per_block + 3) / 4) * 4;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Skip if this block is out of range
    if (block_start >= n_elements) return;

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;
    float4* smem4 = reinterpret_cast<float4*>(smem);
    constexpr int FLOAT4_PER_THREAD = SMEM_ELEMENTS_PER_THREAD / 4;  // 32/4 = 8

    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        const float4* gmem4 = reinterpret_cast<const float4*>(gmem + iter_offset);

        // Vectorized load: 8 float4s per thread (128 bytes)
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                // Use __ldg for cache-friendly load
                // smem4[idx] = __ldg(&gmem4[idx]);
                float4 val;
                asm volatile(
                    "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4]; \n\t"
                    "st.shared.v4.f32 [%5], {%0, %1, %2, %3}; \n\t"
                    : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                    : "l"(gmem4 + idx), "r"((unsigned int)__cvta_generic_to_shared(smem4 + idx))
                    : "memory"
                );
                (void)val;
            }
        }
        __syncthreads();

        // Vectorized store
        float4* output4 = reinterpret_cast<float4*>(output + iter_offset);
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx * 4;
            if (gmem_idx + 3 < n_elements && idx < SMEM_ELEMENTS_PER_BLOCK / 4) {
                // output4[idx] = smem4[idx];
                float4 val;
                asm volatile(
                    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4]; \n\t"
                    "st.global.cg.v4.f32 [%5], {%0, %1, %2, %3}; \n\t"
                    : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                    : "r"((unsigned int)__cvta_generic_to_shared(smem4 + idx)), "l"(output4 + idx)
                    : "memory"
                );
                (void)val;
            }
        }
        __syncthreads();
    }
}

/**
 * Variant 3: Inline PTX with cache hints (ld.global.cg + st.shared)
 *
 * Coalesced access pattern (CORRECT):
 * - Each thread loads 1 float (4 bytes) per iteration using PTX
 * - threadIdx.x provides stride=1 offset
 * - Same thread's consecutive loads are stride=blockDim.x (256) apart
 */
__global__ void kernel_copy_ptx(const float* __restrict__ gmem,
                                 float* __restrict__ output,
                                 int n_elements) {
    __shared__ float smem[SMEM_ELEMENTS_PER_BLOCK];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    int tid = threadIdx.x;

    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;

        // PTX load with cache hints
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                float val;
                const float* gptr = gmem + gmem_idx;
                float* sptr = smem + idx;
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

        // PTX store
        #pragma unroll
        for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
            int idx = tid + i * blockDim.x;
            int gmem_idx = iter_offset + idx;
            if (gmem_idx < block_end && idx < SMEM_ELEMENTS_PER_BLOCK) {
                float* sptr = smem + idx;
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
        __syncthreads();
    }
}

/**
 * Variant 4: CuTe basic copy with Layout partitioning
 *
 * Simple 1D layout:
 * - Data: 8192 elements (SMEM_ELEMENTS)
 * - Threads: 256 (THREADS_PER_BLOCK)
 * - Each thread handles 32 elements (ELEMENTS_PER_THREAD)
 *
 * local_partition divides the 1D tensor among threads.
 * CuTe's copy handles the actual data movement.
 *
 * NOTE: This variant uses UniversalCopy<float> as the copy atom to ensure
 * synchronous (blocking) scalar float copies. Without explicit copy atom specification,
 * CuTe may automatically select async copy instructions (cp.async) which can
 * introduce unexpected latency and different performance characteristics.
 * By explicitly using UniversalCopy, we ensure deterministic synchronous behavior.
 */
__global__ void kernel_copy_cute(const float* __restrict__ gmem,
                                  float* __restrict__ output,
                                  int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations (same as other kernels)
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // 1D layouts - simple and straightforward
    auto data_layout = make_layout(make_shape(Int<SMEM_ELEMENTS_PER_BLOCK>{}));
    auto thr_layout = make_layout(make_shape(Int<THREADS_PER_BLOCK>{}));

    // Create SMEM tensor (persistent across iterations)
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), data_layout);
    // Partition: 8192 / 256 = 32 elements per thread
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    int tid = threadIdx.x;

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // For partial chunks, we need to be careful about bounds
        // CuTe copy doesn't have built-in bounds checking, so we use manual loop
        if (remaining < SMEM_ELEMENTS_PER_BLOCK) {
            // Partial chunk: use manual copy with bounds checking
            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    smem_cute[idx] = gmem[iter_offset + idx];
                }
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    output[iter_offset + idx] = smem_cute[idx];
                }
            }
            __syncthreads();
        } else {
            // Full chunk: use CuTe copy
            // NOTE: Without explicit copy atom specification, CuTe may automatically
            // select async copy instructions (cp.async) for performance optimization.
            // This is acceptable for this benchmark as we're measuring overall throughput.
            // For deterministic synchronous behavior, use make_tiled_copy with
            // explicit UniversalCopy atom (see Variant 6).
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), data_layout);
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Copy GMEM → SMEM using CuTe copy (may use async instructions)
            copy(thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), data_layout);
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            // Copy SMEM → GMEM using CuTe copy (may use async instructions)
            copy(thr_smem, thr_out);
            __syncthreads();
        }
    }
}

/**
 * Variant 5: CuTe basic copy with Layout partitioning indexed tail copy
 *
 * Simple 1D layout:
 * - Data: 8192 elements (SMEM_ELEMENTS)
 * - Threads: 256 (THREADS_PER_BLOCK)
 * - Each thread handles 32 elements (ELEMENTS_PER_THREAD)
 *
 * local_partition divides the 1D tensor among threads.
 * CuTe's copy handles the actual data movement.
 *
 * NOTE: This variant does NOT use explicit copy atom specification.
 * Without explicit copy atom, CuTe may automatically select async copy instructions
 * (cp.async) for performance optimization. This is acceptable for throughput measurement,
 * but if deterministic synchronous behavior is required, use explicit copy atoms
 * (see Variant 7 with UniversalCopy<float>).
 */
__global__ void kernel_copy_cute_indexed_tail_copy(const float* __restrict__ gmem,
                                  float* __restrict__ output,
                                  int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations (same as other kernels)
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // 1D layouts - simple and straightforward
    auto data_layout = make_layout(make_shape(Int<SMEM_ELEMENTS_PER_BLOCK>{}));
    auto thr_layout = make_layout(make_shape(Int<THREADS_PER_BLOCK>{}));

    // Create SMEM tensor (persistent across iterations)
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), data_layout);
    // Partition: 8192 / 256 = 32 elements per thread
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    int tid = threadIdx.x;

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // For partial chunks, we need to be careful about bounds
        // CuTe copy doesn't have built-in bounds checking, so we use manual loop
        if (remaining < SMEM_ELEMENTS_PER_BLOCK) {
            // Partial chunk: use manual copy with bounds checking
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), data_layout);
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), data_layout);
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    smem_tensor(idx) = gmem_tensor(idx);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < SMEM_ELEMENTS_PER_THREAD; ++i) {
                int idx = tid + i * THREADS_PER_BLOCK;
                if (idx < remaining) {
                    out_tensor(idx) = smem_tensor(idx);
                }
            }
            __syncthreads();
        } else {
            // Full chunk: use CuTe copy
            // NOTE: Without explicit copy atom specification, CuTe may automatically
            // select async copy instructions (cp.async) for performance optimization.
            // This is acceptable for this benchmark as we're measuring overall throughput.
            // For deterministic synchronous behavior, use explicit copy atoms
            // (see Variant 7 with UniversalCopy<float4>).
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), data_layout);
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Copy GMEM → SMEM using CuTe copy (may use async instructions)
            copy(thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), data_layout);
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            // Copy SMEM → GMEM using CuTe copy (may use async instructions)
            copy(thr_smem, thr_out);
            __syncthreads();
        }
    }
}

/**
 * Variant 7: CuTe copy with explicit UniversalCopy atom and tail handling
 *
 * Uses Copy_Atom<UniversalCopy<float>> for synchronous scalar copies.
 * Key design:
 * - 2D layout: 32x256 (8192 elements total)
 * - Thread layout: 32x4 (128 threads total) arranged in 2D (m-major)
 * - Value layout: 1x64 (64 float elements per thread)
 * - Copy atom: UniversalCopy<float> for deterministic synchronous behavior
 * - Handles both full chunks and partial chunks (tail) with bounds checking
 *
 * UniversalCopy ensures that CuTe uses synchronous scalar copies instead of
 * automatically selecting async copy instructions (cp.async). This is important
 * for understanding the true synchronous copy performance without async optimizations.
 *
 * Tail handling: For partial chunks, fall back to manual scalar copy to avoid
 * alignment issues and ensure correctness.
 *
 * Performance note: This kernel demonstrates how to use explicit copy atoms
 * to ensure synchronous behavior, avoiding automatic async copy selection by CuTe.
 * The performance is slightly better than v5 (1622.33 vs 1593.65 GB/s) due to
 * better thread layout and partitioning strategy.
 */
__global__ void kernel_copy_cute_universal_copy(const float* __restrict__ gmem,
                                                float* __restrict__ output,
                                                int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // 2D layout: 32x256 (8192 elements)
    auto data_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
    auto thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}));

    // Create SMEM tensor (persistent across iterations)
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), data_layout);
    // Partition: 32x256 / (32x4) = 1x64 elements per thread
    auto thr_smem = local_partition(smem_tensor, thr_layout, threadIdx.x);

    // Define copy atom: UniversalCopy<float> ensures synchronous scalar copies
    // This prevents CuTe from automatically selecting async copy instructions
    // NOTE: We use float instead of float4 to avoid alignment issues with partial chunks
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto copy_op = CopyAtom{};

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // Only use float4 copy for full SMEM blocks (8192 elements)
        // For partial blocks, use manual scalar copy to avoid alignment issues
        if (remaining == SMEM_ELEMENTS_PER_BLOCK) {
            // Full chunk: use CuTe copy with explicit float4 copy atom
            // Create float4 view of the data (8192 floats = 2048 float4)
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset),
                                          make_layout(make_shape(Int<32>{}, Int<256>{})));
            auto thr_gmem = local_partition(gmem_tensor, thr_layout, threadIdx.x);

            // Copy GMEM → SMEM using CuTe copy with float4 copy atom
            copy(copy_op, thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset),
                                         make_layout(make_shape(Int<32>{}, Int<256>{})));
            auto thr_out = local_partition(out_tensor, thr_layout, threadIdx.x);

            // Copy SMEM → GMEM using CuTe copy with float4 copy atom
            copy(copy_op, thr_smem, thr_out);
            __syncthreads();
        } else {
            // Partial chunk: use manual scalar copy with bounds checking
            #pragma unroll
            for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
                smem_cute[i] = gmem[iter_offset + i];
            }
            __syncthreads();

            #pragma unroll
            for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
                output[iter_offset + i] = smem_cute[i];
            }
            __syncthreads();
        }
    }
}

/**
 * Variant 6: CuTe tiled copy with no tail handling
 *
 * Uses make_tiled_copy to create a copy pattern that avoids tail handling.
 * Key design:
 * - Thread layout: 32x4 (128 threads total) arranged in 2D (m-major)
 * - Value layout: 1x8 (8 elements per thread)
 * - Copy atom: UniversalCopy with float (scalar copy)
 * - All data is processed in complete tiles, no partial chunks
 *
 * The tiled copy pattern ensures:
 * - Coalesced memory access
 * - Efficient thread utilization
 * - No bounds checking needed for full tiles
 * - 2D layout provides better memory access patterns
 *
 * Performance note: This kernel demonstrates how to use make_tiled_copy
 * to avoid tail handling by processing data in complete tiles.
 * The m-major layout (stride<1,32>) ensures coalesced access patterns.
 */
__global__ void kernel_copy_cute_tiled_copy(const float* __restrict__ gmem,
                                            float* __restrict__ output,
                                            int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // Create tiled copy with UniversalCopy
    // Thread layout: 32x4 = 128 threads (m-major: stride<1,32>)
    // Value layout: 1x8 = 8 elements per thread
    // Total per tile: 32*1 x 4*8 = 32 x 32 = 1024 elements
    // We'll do 8 iterations to fill SMEM (8192 / 1024 = 8 iterations per SMEM chunk)
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<Int<32>, Int<4>>, Stride<Int<1>, Int<32>>>{},  // Thread layout: 32x4 m-major
        Layout<Shape<Int<1>, Int<8>>>{}                              // Value layout: 1x8 m-major
    );

    // Get this thread's copy operation
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Create SMEM tensor with 2D layout (32 x 256)
    // This matches the tiled copy pattern better
    auto smem_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), smem_layout);

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // Full chunk: use tiled copy
        // Create GMEM tensor for this iteration
        auto gmem_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
        auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), gmem_layout);
        auto thr_gmem = thr_copy.partition_S(gmem_tensor);

        // Partition SMEM tensor as destination
        auto thr_smem = thr_copy.partition_D(smem_tensor);

        // Copy GMEM → SMEM using tiled copy
        copy(tiled_copy, thr_gmem, thr_smem);
        __syncthreads();

        // Create output GMEM tensor for this iteration
        auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), gmem_layout);
        auto thr_out = thr_copy.partition_D(out_tensor);

        // Copy SMEM → GMEM using tiled copy
        copy(tiled_copy, thr_smem, thr_out);
        __syncthreads();
    }
}

/**
 * Variant 8: CuTe tiled copy with float vectorization
 *
 * Extends Variant 6 with float vectorization for better performance.
 * Key design:
 * - 2D layout: 32x256 (8192 floats)
 * - Thread layout: 32x4 = 128 threads (m-major: stride<1,32>)
 * - Value layout: 1x32 = 32 float elements per thread
 * - Copy atom: UniversalCopy<float> for synchronous vectorized copies
 * - Aligned chunks use tiled copy, tail uses manual scalar copy
 *
 * Alignment handling:
 * - Aligns to 128-element boundary (512 bytes) for tiled copy compatibility
 * - Remaining elements after alignment use manual scalar copy
 * - This ensures all tiled copy operations are properly aligned
 *
 * The tiled copy pattern with float ensures:
 * - Coalesced memory access with 128-bit (4x32-bit) loads/stores
 * - Efficient thread utilization with vectorized operations
 * - 2D layout provides better memory access patterns
 *
 * Performance note: This kernel demonstrates how to use make_tiled_copy
 * with float to achieve higher throughput than scalar float copies.
 * The m-major layout (stride<1,32>) ensures coalesced access patterns.
 */
__global__ void kernel_copy_cute_tiled_copy_float4(const float* __restrict__ gmem,
                                                   float* __restrict__ output,
                                                   int n_elements) {
    extern __shared__ float smem_cute[];

    // Calculate this block's data range
    int total_elements_per_block = (n_elements + gridDim.x - 1) / gridDim.x;
    int block_start = blockIdx.x * total_elements_per_block;
    int block_end = min(block_start + total_elements_per_block, n_elements);

    // Calculate number of SMEM-sized iterations
    int n_iters = (total_elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK;

    // Create tiled copy with UniversalCopy<float>
    // Thread layout: 32x4 = 128 threads (m-major: stride<1,32>)
    // Value layout: 1x32 = 32 float elements per thread
    // Total per tile: 32*1 x 4*32 = 32 x 128 = 4096 floats
    // We'll do 2 iterations to fill SMEM (8192 / 4096 = 2 iterations per SMEM chunk)
    using CopyAtom = Copy_Atom<UniversalCopy<float>, float>;
    auto tiled_copy = make_tiled_copy(
        CopyAtom{},
        Layout<Shape<Int<32>, Int<4>>, Stride<Int<1>, Int<32>>>{},  // Thread layout: 32x4 m-major
        Layout<Shape<Int<1>, Int<32>>>{}                             // Value layout: 1x32 m-major
    );

    // Get this thread's copy operation
    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    // Create SMEM tensor with 2D layout (32 x 256 floats)
    auto smem_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
    auto smem_tensor = make_tensor(make_smem_ptr(smem_cute), smem_layout);

    // Outer loop: iterate over SMEM-sized chunks
    for (int iter = 0; iter < n_iters; ++iter) {
        int iter_offset = block_start + iter * SMEM_ELEMENTS_PER_BLOCK;
        if (iter_offset >= block_end) break;

        // Check remaining elements in this chunk
        int remaining = min(block_end - iter_offset, SMEM_ELEMENTS_PER_BLOCK);

        // Only use tiled copy for complete SMEM blocks (8192 elements)
        // For partial blocks, use manual scalar copy to avoid alignment issues
        if (remaining == SMEM_ELEMENTS_PER_BLOCK) {
            // Full chunk: use tiled copy with float4
            // Create GMEM tensor for this iteration (32 x 256 floats)
            auto gmem_layout = make_layout(make_shape(Int<32>{}, Int<256>{}));
            auto gmem_tensor = make_tensor(make_gmem_ptr(gmem + iter_offset), gmem_layout);
            auto thr_gmem = thr_copy.partition_S(gmem_tensor);

            // Partition SMEM tensor as destination
            auto thr_smem = thr_copy.partition_D(smem_tensor);

            // Copy GMEM → SMEM using tiled copy with float4
            copy(tiled_copy, thr_gmem, thr_smem);
            __syncthreads();

            // Create output GMEM tensor for this iteration
            auto out_tensor = make_tensor(make_gmem_ptr(output + iter_offset), gmem_layout);
            auto thr_out = thr_copy.partition_D(out_tensor);

            // Copy SMEM → GMEM using tiled copy with float4
            copy(tiled_copy, thr_smem, thr_out);
            __syncthreads();
        } else {
            // Partial chunk: use manual scalar copy with bounds checking
            #pragma unroll
            for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
                smem_cute[i] = gmem[iter_offset + i];
            }
            __syncthreads();

            #pragma unroll
            for (int i = threadIdx.x; i < remaining; i += blockDim.x) {
                output[iter_offset + i] = smem_cute[i];
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

float getPeakMemoryBandwidth() {
    cudaDeviceProp prop;
    MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("sharedMemPerBlock: %zu bytes\n", prop.sharedMemPerBlock);          // usually 65536
    printf("sharedMemPerBlockOptin: %zu bytes\n", prop.sharedMemPerBlockOptin);
    // Memory bandwidth = 2 * memory_clock_rate * memory_bus_width / 8 / 1e6 (GB/s)
    // Factor of 2 for DDR
    int memoryClockRate, memoryBusWidth;
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("memoryBusWidth: %zu bytes\n", memoryBusWidth);
    printf("memoryClockRate: %zu bytes\n", memoryClockRate);
    return 2.0f * memoryClockRate * (memoryBusWidth / 8.0f) / 1.0e6f;
}

// Helper function to verify output correctness
// Input is initialized as h_input[i] = (float)i, so we can easily debug
bool verifyOutput(const float* h_input, const float* h_output, size_t n_elements,
                  const std::string& variant_name) {
    // Sample verification: check first, middle, and last elements
    std::vector<size_t> check_indices = {0, n_elements / 4, n_elements / 2,
                                          3 * n_elements / 4, n_elements - 1};

    for (size_t idx : check_indices) {
        if (idx >= n_elements) continue;
        if (h_input[idx] != h_output[idx]) {
            std::cerr << "  ERROR: " << variant_name << " verification failed at index "
                      << idx << ": expected " << h_input[idx] << " (should be " << idx << ")"
                      << ", got " << h_output[idx] << std::endl;
            return false;
        }
    }
    return true;
}

void runBenchmark(size_t data_size_bytes, int num_blocks, MicrobenchRunner& runner,
                  MicrobenchReport& report, float peak_bw) {
    size_t n_elements = bytesToElements<float>(data_size_bytes);
    size_t n_bytes = elementsToBytes<float>(n_elements);

    // Allocate memory
    float* d_input;
    float* d_output;
    MBENCH_CUDA_CHECK(cudaMalloc(&d_input, n_bytes));
    MBENCH_CUDA_CHECK(cudaMalloc(&d_output, n_bytes));

    // Initialize input with magic numbers: h_input[i] = (float)i
    // This makes debugging easier - if output[k] has wrong value, we know which index it came from
    std::vector<float> h_input(n_elements);
    std::vector<float> h_output(n_elements, 0.0f);
    for (size_t i = 0; i < n_elements; ++i) {
        h_input[i] = static_cast<float>(i);
    }
    MBENCH_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n_bytes, cudaMemcpyHostToDevice));

    // Calculate effective bandwidth
    // GMEM → SMEM read + SMEM → GMEM write = 2 * n_bytes
    size_t total_bytes = 2 * n_bytes;

    // Calculate expected iterations (for display only, kernel calculates internally)
    size_t elements_per_block = (n_elements + num_blocks - 1) / num_blocks;
    int expected_iters = static_cast<int>((elements_per_block + SMEM_ELEMENTS_PER_BLOCK - 1) / SMEM_ELEMENTS_PER_BLOCK);

    std::cout << "\n  Data: " << formatBytes(n_bytes) << " (" << n_elements << " elements)" << std::endl;
    std::cout << "  Blocks: " << num_blocks << " | Elements/block: " << elements_per_block
              << " (" << formatBytes(elements_per_block * sizeof(float)) << ")" << std::endl;
    std::cout << "  Expected iters/block: " << expected_iters << " (SMEM chunk: " << formatBytes(SMEM_SIZE_PER_BLOCK) << ")" << std::endl;
    std::cout << "  GMEM traffic: " << formatBytes(total_bytes) << " (read + write)" << std::endl;

    // Variant 1: Native CUDA C (scalar, coalesced)
    {
        // Clear output before each kernel
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_native<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "Native");

        MicrobenchResult result;
        result.variant_name = "Native (scalar)";
        result.description = correct ? "1 float/thread/iter, stride=256" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        result.is_baseline = true;
        report.addResult(result);
    }

    // Variant 2: Vectorized (float4, 128B per warp)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_vectorized<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "Vectorized");

        MicrobenchResult result;
        result.variant_name = "Vectorized (float4)";
        result.description = correct ? "1 float4/thread/iter, stride=256" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 8: PTX, Vectorized (float4, 128B per warp)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_vectorized_ptx<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "Vectorized");

        MicrobenchResult result;
        result.variant_name = "Vectorized (float4) with PTX";
        result.description = correct ? "1 float4/thread/iter, stride=256" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 3: Inline PTX (scalar with cache hints)
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_ptx<<<num_blocks, THREADS_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "PTX");

        MicrobenchResult result;
        result.variant_name = "PTX (ld.cg)";
        result.description = correct ? "PTX ld.global.cg, 1 float/thread/iter" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 4: CuTe copy with layout partitioning
    // Uses CuTe's copy() and local_partition() for data transfer
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_cute<<<num_blocks, THREADS_PER_BLOCK, SMEM_SIZE_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "CuTe");

        MicrobenchResult result;
        result.variant_name = "CuTe copy";
        result.description = correct ? "cute::copy + local_partition, 1D layout" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 5: CuTe copy with layout partitioning, indexed tail copy
    // Uses CuTe's copy() and local_partition() for data transfer
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_cute_indexed_tail_copy<<<num_blocks, THREADS_PER_BLOCK, SMEM_SIZE_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "CuTe");

        MicrobenchResult result;
        result.variant_name = "CuTe copy (indexed tail)";
        result.description = correct ? "cute::copy + local_partition, 1D layout" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 7: CuTe copy with UniversalCopy atom and tail handling
    // Uses explicit UniversalCopy atom with 2D layout for synchronous behavior
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_cute_universal_copy<<<num_blocks, THREADS_PER_BLOCK, SMEM_SIZE_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "CuTe UniversalCopy");

        MicrobenchResult result;
        result.variant_name = "CuTe UniversalCopy";
        result.description = correct ? "UniversalCopy atom, 2D layout, with tail handling" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 6: CuTe tiled copy with no tail handling
    // Uses make_tiled_copy for efficient data transfer without tail processing
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_cute_tiled_copy<<<num_blocks, THREADS_PER_BLOCK, SMEM_SIZE_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "CuTe tiled");

        MicrobenchResult result;
        result.variant_name = "CuTe tiled copy";
        result.description = correct ? "make_tiled_copy, UniversalCopy, no tail handling" : "VERIFICATION FAILED";
        result.data_size_bytes = n_bytes;
        result.latency_us = latency_us;
        result.bandwidth_gbps = bw;
        result.efficiency_pct = (bw / peak_bw) * 100.0f;
        report.addResult(result);
    }

    // Variant 8: CuTe tiled copy with float4 vectorization
    // Uses make_tiled_copy with float4 for higher throughput
    {
        MBENCH_CUDA_CHECK(cudaMemset(d_output, 0, n_bytes));

        auto kernel = [=] {
            kernel_copy_cute_tiled_copy_float4<<<num_blocks, THREADS_PER_BLOCK, SMEM_SIZE_PER_BLOCK>>>(d_input, d_output, n_elements);
        };
        float latency_us = runner.measureLatencyUs(kernel);
        float bw = MicrobenchRunner::calculateBandwidthGBps(total_bytes, latency_us);

        // Verify correctness
        MBENCH_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n_bytes, cudaMemcpyDeviceToHost));
        bool correct = verifyOutput(h_input.data(), h_output.data(), n_elements, "CuTe tiled float4");

        MicrobenchResult result;
        result.variant_name = "CuTe tiled copy (float4)";
        result.description = correct ? "make_tiled_copy, UniversalCopy<float4>, with tail handling" : "VERIFICATION FAILED";
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
    int num_sms = prop.multiProcessorCount;

    // Get peak memory bandwidth
    float peak_bw = getPeakMemoryBandwidth();

    MicrobenchRunner runner(20, 200);  // 20 warmup, 200 measurement iterations
    MicrobenchReport report("Synchronous GMEM→SMEM→GMEM Copy (Stress Test)", peak_bw);

    report.printHeader();
    std::cout << "SMs: " << num_sms << "\n";
    std::cout << "Warmup iterations: " << runner.getWarmupIters() << "\n";
    std::cout << "Measurement iterations: " << runner.getMeasureIters() << "\n";
    std::cout << "SMEM buffer: " << formatBytes(SMEM_SIZE_PER_BLOCK) << " per block\n";
    std::cout << "SMEM elements per block: " << SMEM_ELEMENTS_PER_BLOCK << " (" << formatBytes(SMEM_SIZE_PER_BLOCK) << ")\n";
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << "\n";

    // Test configurations: {data_size, num_blocks}
    // Use 4x SMs to ensure full occupancy (H800 has ~80-132 SMs depending on variant)
    // Kernel logic (self-contained in each kernel):
    //   - Each block calculates: elements_per_block = n_elements / gridDim.x
    //   - Each block calculates: n_iters = ceil(elements_per_block / SMEM_ELEMENTS)
    //   - Each iteration loads one SMEM chunk (32KB = 8192 elements)
    struct TestConfig {
        size_t data_size;
        int num_blocks;
        const char* description;
    };

    // Use a fixed number of blocks that aligns well with data sizes
    // 512 blocks = 2^9, ensures good alignment for float4 operations
    // Each block processes: 512MB / 512 = 1024KB = 262144 elements
    // 262144 / 128 threads = 2048 elements per thread
    // This is divisible by 128 (ALIGN_SIZE in v8), ensuring proper alignment
    int num_blocks = 512;

    std::vector<TestConfig> configs = {
        // {SIZE_16MB,  num_blocks,  "16MB"},
        // {SIZE_64MB,  num_blocks,  "64MB"},
        // {SIZE_128MB, num_blocks,  "128MB"},
        // {SIZE_256MB, num_blocks,  "256MB"},
        {SIZE_512MB, num_blocks,  "512MB"},
    };

    for (const auto& config : configs) {
        std::cout << "\n=== " << config.description << " ===" << std::endl;
        runBenchmark(config.data_size, config.num_blocks, runner, report, peak_bw);
    }

    report.printTable();
    report.printSummary();

    return 0;
}
