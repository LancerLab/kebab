#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace baseline {

/**
 * @brief Hand-optimized CUDA baseline for element-wise addition: C = A + B
 * 
 * This implementation uses the same API signature as the CuTe version
 * for fair performance comparison.
 * 
 * Optimizations:
 * - Vectorized memory access (float4 for float, half2 for half)
 * - Grid-stride loops for better scalability
 * - Proper block sizing (256 threads per block)
 * - Memory coalescing for maximum bandwidth utilization
 * 
 * @tparam T Data type (float or __half)
 * @param A Input array A
 * @param B Input array B
 * @param C Output array C
 * @param N Number of elements
 * @param stream CUDA stream for asynchronous execution (default: 0)
 */
void elementwise_add(const float* A, const float* B, float* C, int N, cudaStream_t stream = 0);
void elementwise_add(const __half* A, const __half* B, __half* C, int N, cudaStream_t stream = 0);

} // namespace baseline
