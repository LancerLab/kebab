/**
 * @file cublas_reference.h
 * @brief Shared cuBLAS reference implementation for verification
 *
 * This header provides common cuBLAS reference functions used by both
 * bench_gemm and runonce_gemm to ensure consistent verification.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <iostream>
#include <cmath>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

namespace kebab {
namespace utils {

/**
 * @brief cuBLAS configuration for different matrix storage formats
 */
struct CublasGemmConfig {
    cublasOperation_t opA;
    cublasOperation_t opB;
    int ldA;
    int ldB;
    int ldC;
};

/**
 * @brief Get cuBLAS configuration based on matrix storage format
 *
 * @param lhs_format 'R' for row-major, 'C' for column-major (A matrix)
 * @param rhs_format 'R' for row-major, 'C' for column-major (B matrix)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @return CublasGemmConfig Configuration for cuBLAS call
 */
inline CublasGemmConfig getCublasGemmConfig(char lhs_format, char rhs_format, int M, int N, int K) {
    CublasGemmConfig config;
    config.ldC = M;  // C is always column-major output

    if (lhs_format == 'R' && rhs_format == 'R') {
        // RR: Both row-major
        config.opA = CUBLAS_OP_T;
        config.opB = CUBLAS_OP_T;
        config.ldA = K;
        config.ldB = N;
    } else if (lhs_format == 'R' && rhs_format == 'C') {
        // RC: A row-major, B column-major
        config.opA = CUBLAS_OP_T;
        config.opB = CUBLAS_OP_N;
        config.ldA = K;
        config.ldB = K;
    } else if (lhs_format == 'C' && rhs_format == 'R') {
        // CR: A column-major, B row-major
        config.opA = CUBLAS_OP_N;
        config.opB = CUBLAS_OP_T;
        config.ldA = M;
        config.ldB = N;
    } else {  // CC
        // CC: Both column-major
        config.opA = CUBLAS_OP_N;
        config.opB = CUBLAS_OP_N;
        config.ldA = M;
        config.ldB = K;
    }

    return config;
}

/**
 * @brief Get verification tolerance based on data type and matrix size
 *
 * Uses size-dependent tolerance for half precision due to accumulation errors.
 *
 * @tparam T Data type (float, __half, __nv_bfloat16)
 * @param size Matrix size (M=N=K for square matrices)
 * @return float Tolerance value for verification
 */
template<typename T>
inline float getVerificationTolerance(int size) {
    if constexpr (std::is_same_v<T, float>) {
        return 1e-3f;
    } else if constexpr (std::is_same_v<T, __half>) {
        // Size-dependent tolerance for half precision
        if (size <= 512) return 0.15f;
        else if (size <= 1024) return 0.25f;
        else return 2.0f;  // Larger matrices need more tolerance due to FP16 accumulation
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        // BFloat16 has lower precision, needs higher tolerance
        if (size <= 512) return 0.2f;
        else if (size <= 1024) return 0.35f;
        else return 2.0f;
    }
    return 1e-3f;  // Default
}

/**
 * @brief Run cuBLAS reference GEMM
 *
 * @tparam T Data type (float, __half, __nv_bfloat16)
 * @param handle cuBLAS handle
 * @param d_A Device pointer to A matrix
 * @param d_B Device pointer to B matrix
 * @param d_C Device pointer to C matrix (output)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param config cuBLAS configuration
 */
template<typename T>
inline void runCublasGemm(cublasHandle_t handle, const T* d_A, const T* d_B, T* d_C,
                          int M, int N, int K, const CublasGemmConfig& config) {
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, config.opA, config.opB, M, N, K,
                    &alpha, d_A, config.ldA, d_B, config.ldB, &beta, d_C, config.ldC);
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle, config.opA, config.opB, M, N, K,
                    &alpha, d_A, config.ldA, d_B, config.ldB, &beta, d_C, config.ldC);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, config.opA, config.opB, M, N, K,
                     &alpha, d_A, CUDA_R_16BF, config.ldA,
                     d_B, CUDA_R_16BF, config.ldB,
                     &beta, d_C, CUDA_R_16BF, config.ldC,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
}

/**
 * @brief Verify GEMM result against cuBLAS reference
 *
 * @tparam T Data type (float, __half, __nv_bfloat16)
 * @param d_A Device pointer to A matrix
 * @param d_B Device pointer to B matrix
 * @param d_C_test Device pointer to test result
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param lhs_format 'R' for row-major, 'C' for column-major (A matrix)
 * @param rhs_format 'R' for row-major, 'C' for column-major (B matrix)
 * @param tolerance Tolerance for comparison (use getVerificationTolerance for default)
 * @param verbose Print detailed mismatch information
 * @return true if verification passed
 */
template<typename T>
inline bool verifyCublasGemm(const T* d_A, const T* d_B, const T* d_C_test,
                              int M, int N, int K,
                              char lhs_format, char rhs_format,
                              float tolerance, bool verbose = false) {
    // Allocate reference result
    T* d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_C_ref, 0, M * N * sizeof(T)));

    // Run cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    auto config = getCublasGemmConfig(lhs_format, rhs_format, M, N, K);
    runCublasGemm(handle, d_A, d_B, d_C_ref, M, N, K, config);
    CUDA_CHECK(cudaDeviceSynchronize());
    cublasDestroy(handle);

    // Copy to host for comparison
    std::vector<T> h_C_test(M * N), h_C_ref(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_test.data(), d_C_test, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_C_ref));

    // Compare results
    int mismatch_count = 0;
    float max_diff = 0.0f;
    int first_mismatch_idx = -1;

    for (int i = 0; i < M * N; ++i) {
        float val = static_cast<float>(h_C_test[i]);
        float ref = static_cast<float>(h_C_ref[i]);
        float diff = std::abs(val - ref);
        max_diff = std::max(max_diff, diff);
        if (diff > tolerance) {
            if (first_mismatch_idx < 0) first_mismatch_idx = i;
            mismatch_count++;
        }
    }

    bool passed = (mismatch_count == 0);

    if (verbose || !passed) {
        std::cout << "\nVerification: " << (passed ? "PASS ✓" : "FAIL ✗") << std::endl;
        std::cout << "  Max diff: " << max_diff << ", Tolerance: " << tolerance
                  << ", Mismatches: " << mismatch_count << "/" << (M * N) << std::endl;
        if (!passed && first_mismatch_idx >= 0) {
            std::cout << "  First mismatch at index " << first_mismatch_idx
                      << ": test=" << static_cast<float>(h_C_test[first_mismatch_idx])
                      << ", ref=" << static_cast<float>(h_C_ref[first_mismatch_idx]) << std::endl;
        }
    }

    return passed;
}

} // namespace utils
} // namespace kebab

