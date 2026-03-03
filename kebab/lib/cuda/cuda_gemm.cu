/**
 * @file cuda_gemm.cu
 * @brief CUDA GEMM dispatch interface
 *
 * This file provides the public GEMM interface and dispatches to
 * specific kernel implementations based on version:
 *   - V1: Warp Tiling (cuda_gemm_v1_warptiling_baseline.cu)
 *   - V2: WGMMA + TMA (future)
 */

#include "kebab/cuda/cuda_gemm.h"
#include "kebab/cuda/cuda_kernel_utils.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace baseline {

namespace {

enum class V20DecompositionMode {
    DataParallel,
    PersistentHilbert,
    Heuristic,
};

V20DecompositionMode get_v20_mode() {
    const char* env = std::getenv("KEBAB_V20_MODE");
    if (env == nullptr) {
        return V20DecompositionMode::Heuristic;
    }
    std::string mode(env);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    if (mode == "DP" || mode == "DATAPARALLEL" || mode == "V15") {
        return V20DecompositionMode::DataParallel;
    }
    if (mode == "PERSISTENT" || mode == "HILBERT" || mode == "V19") {
        return V20DecompositionMode::PersistentHilbert;
    }
    return V20DecompositionMode::Heuristic;
}

bool v20_choose_v19_heuristic(int M, int N, int K) {
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int NUM_SM = 128;

    if ((M % BM) != 0 || (N % BN) != 0 || (K % 64) != 0) {
        return false;
    }

    int tiles_m = M / BM;
    int tiles_n = N / BN;
    int total_tiles = tiles_m * tiles_n;
    int tail_tiles = total_tiles % NUM_SM;

    if (total_tiles < (NUM_SM / 2)) {
        return true;
    }
    if (total_tiles <= NUM_SM) {
        return false;
    }
    if (tail_tiles == 0) {
        return false;
    }

    float tail_ratio = static_cast<float>(tail_tiles) / static_cast<float>(NUM_SM);
    bool small_tail_wave = tail_ratio <= 0.35f;
    bool shallow_problem = total_tiles <= 2 * NUM_SM;
    bool long_k = K >= 2048;
    return (small_tail_wave && long_k) || shallow_problem;
}

} // namespace

const char* gemm_cuda_version_feature_name(int version) {
    switch (version) {
        case 1:  return "warptiling_baseline";
        case 2:  return "wgmma_tma";
        case 3:  return "wgmma_tma_warpgroup";
        case 4:  return "wgmma_tma_warpgroup_warpspecialized";
        case 5:  return "wgmma_tma_warpgroup_warpspecialized_persistent";
        case 6:  return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler";
        case 7:  return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d";
        case 8:  return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast";
        case 9:  return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_streamstore";
        case 10: return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore";
        case 11: return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore_hilbert";
        case 12: return "wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore_hilbert_stmatrix_padded";
        case 13: return "wgmma_tma_warpspecialized_persistent_tilescheduler_ptxbarrier_tma2d";
        case 14: return "wgmma_tma_warpspecialized_persistent_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster";
        case 15: return "wgmma_tma_warpspecialized_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_nopersistent";
        case 16: return "wgmma_tma_warpspecialized_persistent_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_linearschedule";
        case 17: return "wgmma_tma_warpgroup_ptxbarrier";
        case 18: return "wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix";
        case 19: return "wgmma_tma_warpgroup_warpspecialized_persistent_hilbert";
        case 20: return "decomposition_heuristic_v15_v19";
        default: return "unknown";
    }
}

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
            gemm_v1_warptiling_fp16_baseline(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 2:
            // V2: WGMMA + TMA (SM90 Hopper, RC mode only)
            // Uses B128 swizzle for optimal performance (92% of cuBLAS for FP16)
            gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B128, cuda_kernel::WGMMA_Swizzle::B128, __half>(
                A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 3:
            // V3: Warp Group with larger tiles (SM90 Hopper, RC mode only)
            gemm_v3_wgmma_tma_warpgroup_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 4:
            // V4: Warp Specialization with multi-stage pipeline (SM90 Hopper, RC mode only)
            gemm_v4_wgmma_tma_warpgroup_warpspecialized_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 5:
            // V5: Larger tiles + register optimization (SM90 Hopper, RC mode only)
            gemm_v5_wgmma_tma_warpgroup_warpspecialized_persistent_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 6:
            // V6: Persistent kernel + tile scheduling (SM90 Hopper, RC mode only)
            gemm_v6_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 7:
            // V7: PTX barriers + 5D TMA (SM90 Hopper, RC mode only)
            gemm_v7_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 8:
            // V8: Thread Block Clusters + TMA Multicast (SM90 Hopper, RC mode only)
            gemm_v8_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 9:
            // V9: Streaming Stores + Clusters + TMA Multicast (SM90 Hopper, RC mode only)
            gemm_v9_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_streamstore_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 10:
            // V10: TMA Async Stores + Clusters + TMA Multicast (SM90 Hopper, RC mode only)
            gemm_v10_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 11:
            // V11: Hilbert Curve Scheduling + TMA Stores (SM90 Hopper, RC mode only)
            gemm_v11_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore_hilbert_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 12:
            // V12: stmatrix + Padded TMA Stores (SM90 Hopper, RC mode only)
            gemm_v12_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore_hilbert_stmatrix_padded_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 13:
            // V13: PTX barriers + 2D TMA (SM90 Hopper, RC mode only)
            gemm_v13_wgmma_tma_warpspecialized_persistent_tilescheduler_ptxbarrier_tma2d_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 14:
            // V14: stmatrix + padded TMA, no cluster/multicast (SM90 Hopper, RC mode only)
            gemm_v14_wgmma_tma_warpspecialized_persistent_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 15:
            // V15: stmatrix + padded TMA, no cluster and no persistent scheduling (SM90 Hopper, RC mode only)
            gemm_v15_wgmma_tma_warpspecialized_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_nopersistent_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 16:
            // V16: stmatrix + padded TMA, no cluster, linear persistent scheduling (SM90 Hopper, RC mode only)
            gemm_v16_wgmma_tma_warpspecialized_persistent_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_linearschedule_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 17:
            // V17: V3 warpgroup + PTX mbarrier sync (SM90 Hopper, RC mode only)
            gemm_v17_wgmma_tma_warpgroup_ptxbarrier_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 18:
            // V18: V5 + stmatrix epilogue (SM90 Hopper, RC mode only)
            gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 19:
            // V19: V5 + Hilbert scheduling (SM90 Hopper, RC mode only)
            gemm_v19_wgmma_tma_warpgroup_warpspecialized_persistent_hilbert_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 20: {
            // V20: decomposition scheduler (Heuristic/DataParallel/PersistentHilbert)
            // Combines existing kernels: v15 (data-parallel) and v19 (persistent hilbert)
            V20DecompositionMode mode = get_v20_mode();
            bool use_v19 = false;
            if (mode == V20DecompositionMode::PersistentHilbert) {
                use_v19 = true;
            } else if (mode == V20DecompositionMode::Heuristic) {
                use_v19 = v20_choose_v19_heuristic(M, N, K);
            }

            if (use_v19) {
                gemm_v19_wgmma_tma_warpgroup_warpspecialized_persistent_hilbert_fp16(
                    A, B, C, M, N, K, lhs_format, rhs_format, stream);
            } else {
                gemm_v15_wgmma_tma_warpspecialized_ptxbarrier_tmastore_hilbert_stmatrix_padded_nocluster_nopersistent_fp16(
                    A, B, C, M, N, K, lhs_format, rhs_format, stream);
            }
            break;
        }
        default:
            fprintf(stderr, "ERROR: Unsupported CUDA version %d\n", version);
            fprintf(stderr, "       Available: 1-20\n");
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
    fprintf(stderr, "ERROR: Float precision not implemented. Use half or bfloat16 precision.\n");
}

// ============================================================================
// BFloat16 GEMM Dispatch
// ============================================================================

void gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
          int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Validate inputs
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer passed to baseline::gemm (bfloat16)\n");
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

    // Version dispatch for BFloat16
    switch (version) {
        case 1:
            gemm_v1_warptiling_bf16_baseline(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 2:
            // V2: WGMMA + TMA (SM90 Hopper, RC mode only)
            // Uses B128 swizzle for optimal performance (65% of cuBLAS for BF16)
            gemm_v2_wgmma_tma<cuda_kernel::WGMMA_Swizzle::B128, cuda_kernel::WGMMA_Swizzle::B128, __nv_bfloat16>(
                A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 3:
            gemm_v3_wgmma_tma_warpgroup_bf16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        // case 11:
        //     gemm_v11_hilbert_bf16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
        //     break;
        case 12:
            gemm_v12_wgmma_tma_warpgroup_warpspecialized_persistent_tilescheduler_ptxbarrier_tma5d_cluster_multicast_tmastore_hilbert_stmatrix_padded_bf16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        default:
            fprintf(stderr, "ERROR: Unsupported CUDA version %d for bfloat16\n", version);
            fprintf(stderr, "       Available: 1, 2, 3, 11, 12\n");
            return;
    }
}

} // namespace baseline
