/**
 * @file cuda_gemm.cu
 * @brief CUDA GEMM dispatch interface
 *
 * This file provides the public GEMM interface and dispatches to
 * specific kernel implementations based on version:
 *   - V1: Warp Tiling (cuda_gemm_v1_warptiling.cu)
 *   - V2: WGMMA + TMA (future)
 */

#include "kebab/cuda/cuda_gemm.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cctype>

namespace baseline {

// ============================================================================
// FP16 GEMM Dispatch
// ============================================================================

void gemm(const __half* A, const __half* B, __half* C,
          int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (half)\n");
        return;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "ERROR: Invalid dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }

    // Parse opmode
    std::string opmode_str(opmode ? opmode : "RC");
    std::transform(opmode_str.begin(), opmode_str.end(), opmode_str.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    char lhs_format = (opmode_str.length() >= 1) ? opmode_str[0] : 'R';
    char rhs_format = (opmode_str.length() >= 2) ? opmode_str[1] : 'C';

    // Version dispatch
    switch (version) {
        case 1:
            // V1: Warp Tiling (RC mode only)
            gemm_v1_warptiling_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 2:
            // V2: WGMMA + TMA (SM90 Hopper, RC mode only)
            gemm_v2_wgmma_tma_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 3:
            // V3: Warp Group with larger tiles (SM90 Hopper, RC mode only)
            gemm_v3_warpgroup_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 4:
            // V4: Warp Specialization with multi-stage pipeline (SM90 Hopper, RC mode only)
            gemm_v4_warpspec_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 5:
            // V5: Larger tiles + register optimization (SM90 Hopper, RC mode only)
            gemm_v5_persistent_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 6:
            // V6: Persistent kernel + tile scheduling (SM90 Hopper, RC mode only)
            gemm_v6_ptxbarrier_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 7:
            // V7: PTX barriers + 5D TMA (SM90 Hopper, RC mode only)
            gemm_v7_cluster_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 8:
            // V8: Thread Block Clusters + TMA Multicast (SM90 Hopper, RC mode only)
            gemm_v8_multicast_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 9:
            // V9: Streaming Stores + Clusters + TMA Multicast (SM90 Hopper, RC mode only)
            gemm_v9_streamstore_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 10:
            // V10: TMA Async Stores + Clusters + TMA Multicast (SM90 Hopper, RC mode only)
            gemm_v10_tmastore_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        default:
            fprintf(stderr, "ERROR: Unsupported CUDA version %d\n", version);
            fprintf(stderr, "       Available: 1-10\n");
            return;
    }
}

// ============================================================================
// FP32 GEMM Dispatch (Not Implemented)
// ============================================================================

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (float)\n");
        return;
    }
    fprintf(stderr, "ERROR: Float precision not implemented. Use half precision.\n");
}

} // namespace baseline