/**
 * @file gemm.cu
 * @brief CuTe GEMM implementation using WGMMA (Hopper SM90+)
 * 
 * This file provides the public GEMM interface and dispatches to the
 * WGMMA implementation for all data types on SM90+ GPUs.
 * 
 * All actual computation is done using CuTe's WGMMA atoms.
 */

#include "kebab/cute/gemm.h"
#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace kebab {
namespace cute {

// Static flag to check device capability only once
static bool g_device_checked = false;
static bool g_device_valid = false;

/**
 * @brief Conversion kernels for FP32 <-> FP16
 */
__global__ void convert_fp32_to_fp16(const float* in, __half* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void convert_fp16_to_fp32(const __half* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

/**
 * @brief FP32 GEMM - converts to FP16 and uses WGMMA
 */
template <>
void gemm<float>(const float* A, const float* B, float* C,
                 int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Check device capability only once (avoid repeated cudaGetDevice/cudaGetDeviceProperties calls)
    if (!g_device_checked) {
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (prop.major < 9) {
            fprintf(stderr, "ERROR: This implementation requires SM90+ (Hopper) GPU\n");
            fprintf(stderr, "       Detected: SM%d.%d\n", prop.major, prop.minor);
            g_device_valid = false;
        } else {
            g_device_valid = true;
        }
        g_device_checked = true;
    }

    if (!g_device_valid) {
        exit(EXIT_FAILURE);
    }
    
    // Allocate FP16 buffers
    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    size_t size_C = M * N * sizeof(__half);
    
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CUDA_CHECK(cudaMalloc(&d_A_fp16, size_A));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, size_B));
    CUDA_CHECK(cudaMalloc(&d_C_fp16, size_C));
    
    // Convert FP32 to FP16
    int threads = 256;
    int total_A = M * K;
    int total_B = K * N;
    int total_C = M * N;
    
    convert_fp32_to_fp16<<<(total_A + threads - 1) / threads, threads, 0, stream>>>(A, d_A_fp16, total_A);
    convert_fp32_to_fp16<<<(total_B + threads - 1) / threads, threads, 0, stream>>>(B, d_B_fp16, total_B);
    
    // Parse opmode: <LHS_format><RHS_format> where format is R (row-major) or C (column-major)
    std::string opmode_str(opmode);
    std::transform(opmode_str.begin(), opmode_str.end(), opmode_str.begin(), 
                   [](unsigned char c){ return std::toupper(c); });
    
    char lhs_format = (opmode_str.length() >= 1) ? opmode_str[0] : 'R';
    char rhs_format = (opmode_str.length() >= 2) ? opmode_str[1] : 'R';
    
    // Version dispatch
    switch (version) {
        case 1:
            // Version 1: WGMMA without TMA with configurable tile sizes
            gemm_wgmma_fp16(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 2:
            // Version 2: WGMMA with TMA with configurable tile sizes
            gemm_wgmma_tma_fp16(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K, lhs_format, rhs_format, stream);
            break;
        default:
            fprintf(stderr, "ERROR: Unsupported CuTe version %d for float\n", version);
            fprintf(stderr, "       Available versions: 1 (WGMMA without TMA), 2 (WGMMA with TMA)\n");
            exit(EXIT_FAILURE);
    }
    
    // Convert back to FP32
    convert_fp16_to_fp32<<<(total_C + threads - 1) / threads, threads, 0, stream>>>(d_C_fp16, C, total_C);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C_fp16));
}

/**
 * @brief FP16 GEMM - direct WGMMA
 */
template <>
void gemm<__half>(const __half* A, const __half* B, __half* C,
                  int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Check device capability only once (avoid repeated cudaGetDevice/cudaGetDeviceProperties calls)
    if (!g_device_checked) {
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        if (prop.major < 9) {
            fprintf(stderr, "ERROR: This implementation requires SM90+ (Hopper) GPU\n");
            fprintf(stderr, "       Detected: SM%d.%d\n", prop.major, prop.minor);
            g_device_valid = false;
        } else {
            g_device_valid = true;
        }
        g_device_checked = true;
    }

    if (!g_device_valid) {
        exit(EXIT_FAILURE);
    }

    // Parse opmode: <LHS_format><RHS_format> where format is R (row-major) or C (column-major)
    std::string opmode_str(opmode);
    std::transform(opmode_str.begin(), opmode_str.end(), opmode_str.begin(),
                   [](unsigned char c){ return std::toupper(c); });

    char lhs_format = (opmode_str.length() >= 1) ? opmode_str[0] : 'R';
    char rhs_format = (opmode_str.length() >= 2) ? opmode_str[1] : 'R';

    // Version dispatch
    switch (version) {
        case 1:
            // Version 1: WGMMA without TMA with configurable tile sizes
            gemm_wgmma_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        case 2:
            // Version 2: WGMMA with TMA with configurable tile sizes
            gemm_wgmma_tma_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
            break;
        default:
            fprintf(stderr, "ERROR: Unsupported CuTe version %d for half\n", version);
            fprintf(stderr, "       Available versions: 1 (WGMMA without TMA), 2 (WGMMA with TMA)\n");
            exit(EXIT_FAILURE);
    }
}

} // namespace cute
} // namespace kebab
