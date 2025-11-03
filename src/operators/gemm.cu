/**
 * @file gemm.cu
 * @brief CuTe GEMM implementation using WGMMA (Hopper SM90+)
 * 
 * This file provides the public GEMM interface and dispatches to the
 * WGMMA implementation for all data types on SM90+ GPUs.
 * 
 * All actual computation is done using CuTe's WGMMA atoms.
 */

#include "cutekernellib/operators/gemm.h"
#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>

namespace cutekernellib {

// Forward declare WGMMA implementation
void gemm_wgmma_fp16_dispatch(const void* A, const void* B, void* C,
                              int M, int N, int K, 
                              char lhs_format, char rhs_format,
                              cudaStream_t stream);

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
template<>
void gemm<float>(const float* A, const float* B, float* C, 
                 int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Check device capability
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: This implementation requires SM90+ (Hopper) GPU\n");
        fprintf(stderr, "       Detected: SM%d.%d\n", prop.major, prop.minor);
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
    
    // Run WGMMA with dispatch
    gemm_wgmma_fp16_dispatch(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K, lhs_format, rhs_format, stream);
    
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
template<>
void gemm<__half>(const __half* A, const __half* B, __half* C, 
                  int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
    // Check device capability
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    if (prop.major < 9) {
        fprintf(stderr, "ERROR: This implementation requires SM90+ (Hopper) GPU\n");
        fprintf(stderr, "       Detected: SM%d.%d\n", prop.major, prop.minor);
        exit(EXIT_FAILURE);
    }
    
    // Parse opmode: <LHS_format><RHS_format> where format is R (row-major) or C (column-major)
    std::string opmode_str(opmode);
    std::transform(opmode_str.begin(), opmode_str.end(), opmode_str.begin(), 
                   [](unsigned char c){ return std::toupper(c); });
    
    char lhs_format = (opmode_str.length() >= 1) ? opmode_str[0] : 'R';
    char rhs_format = (opmode_str.length() >= 2) ? opmode_str[1] : 'R';
    
    // Direct WGMMA call with dispatch
    gemm_wgmma_fp16_dispatch(A, B, C, M, N, K, lhs_format, rhs_format, stream);
}

/**
 * @brief Scaled GEMM - not yet implemented
 */
template<>
void gemm_scaled<float>(const float* A, const float* B, float* C, 
                        int M, int N, int K, float alpha, float beta, 
                        const char* opmode, int version, cudaStream_t stream) {
    if (alpha != 1.0f || beta != 0.0f) {
        fprintf(stderr, "ERROR: Scaled GEMM not yet implemented\n");
        fprintf(stderr, "       Only alpha=1.0, beta=0.0 supported\n");
        exit(EXIT_FAILURE);
    }
    gemm<float>(A, B, C, M, N, K, opmode, version, stream);
}

template<>
void gemm_scaled<__half>(const __half* A, const __half* B, __half* C, 
                         int M, int N, int K, __half alpha, __half beta, 
                         const char* opmode, int version, cudaStream_t stream) {
    float alpha_f = __half2float(alpha);
    float beta_f = __half2float(beta);
    
    if (alpha_f != 1.0f || beta_f != 0.0f) {
        fprintf(stderr, "ERROR: Scaled GEMM not yet implemented\n");
        fprintf(stderr, "       Only alpha=1.0, beta=0.0 supported\n");
        exit(EXIT_FAILURE);
    }
    gemm<__half>(A, B, C, M, N, K, opmode, version, stream);
}

} // namespace cutekernellib
