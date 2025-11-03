/**
 * @file gemm_wgmma_tma.cu
 * @brief GEMM using Hopper WGMMA with TMA (Tensor Memory Accelerator)
 * 
 * Version 2: WGMMA with TMA for improved memory bandwidth
 * 
 * TMA (Tensor Memory Accelerator) is a Hopper feature that:
 * - Offloads data movement from threads to dedicated hardware
 * - Supports multi-dimensional tensor addressing
 * - Provides better memory bandwidth utilization
 * - Reduces register pressure
 */

#include "cutekernellib/operators/gemm.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include "cutlass/cluster_launch.hpp"

using namespace cute;

namespace cutekernellib {

// Forward declare version 1 implementation
void gemm_wgmma_fp16_dispatch(const void* A, const void* B, void* C,
                              int M, int N, int K, 
                              char lhs_format, char rhs_format,
                              cudaStream_t stream);

/**
 * @brief WGMMA with TMA dispatch (Version 2)
 * 
 * For now, this falls back to version 1 until TMA implementation is complete
 */
void gemm_wgmma_tma_fp16_dispatch(const void* A, const void* B, void* C,
                                  int M, int N, int K, 
                                  char lhs_format, char rhs_format,
                                  cudaStream_t stream)
{
    // TODO: Implement TMA version
    // For now, fall back to version 1
    fprintf(stderr, "WARNING: TMA version not yet implemented, falling back to version 1\n");
    gemm_wgmma_fp16_dispatch(A, B, C, M, N, K, lhs_format, rhs_format, stream);
}

} // namespace cutekernellib
