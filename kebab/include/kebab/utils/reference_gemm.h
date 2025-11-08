/**
 * @file reference_gemm.h
 * @brief C++ reference GEMM implementation for verification
 *
 * This header provides a pure C++ reference GEMM implementation that:
 * 1. Uses simple loops for correctness (not performance)
 * 2. Handles different matrix storage formats (row-major and column-major)
 * 3. Can be used to verify cuBLAS calls
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace kebab {
namespace utils {

/**
 * @brief C++ reference GEMM implementation
 *
 * Computes C = A * B using simple loops.
 * Handles different matrix storage formats without data movement.
 *
 * @tparam T Data type (float or double)
 * @param h_A Host pointer to matrix A (M x K logical)
 * @param h_B Host pointer to matrix B (K x N logical)
 * @param h_C Host pointer to matrix C (M x N logical, output)
 * @param M Number of rows in logical A and C
 * @param N Number of columns in logical B and C
 * @param K Number of columns in logical A and rows in logical B
 * @param lhs_format 'R' for row-major, 'C' for column-major (storage format of A)
 * @param rhs_format 'R' for row-major, 'C' for column-major (storage format of B)
 * @param verbose Print debug information
 */
template<typename T>
void referenceGemmCpp(const T* h_A, const T* h_B, T* h_C,
                      int M, int N, int K,
                      char lhs_format = 'C', char rhs_format = 'R',
                      bool verbose = false) {
    // Helper lambda to access matrix element based on storage format
    auto getA = [&](int i, int j) -> T {
        // A is M x K logical
        if (lhs_format == 'R') {
            // Row-major: A[i][j] = A[i*K + j]
            return h_A[i * K + j];
        } else {
            // Column-major: A[i][j] = A[j*M + i]
            return h_A[j * M + i];
        }
    };

    auto getB = [&](int i, int j) -> T {
        // B is K x N logical
        if (rhs_format == 'R') {
            // Row-major: B[i][j] = B[i*N + j]
            return h_B[i * N + j];
        } else {
            // Column-major: B[i][j] = B[j*K + i]
            return h_B[j * K + i];
        }
    };

    auto setC = [&](int i, int j, T val) {
        // C is always M x N in column-major (BLAS standard)
        h_C[j * M + i] = val;
    };

    if (verbose) {
        printf("=== C++ Reference GEMM ===\n");
        printf("M=%d, N=%d, K=%d\n", M, N, K);
        printf("A format: %s, B format: %s\n",
               lhs_format == 'R' ? "row-major" : "col-major",
               rhs_format == 'R' ? "row-major" : "col-major");
    }

    // Compute C = A * B
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = T(0);
            for (int k = 0; k < K; ++k) {
                sum += getA(i, k) * getB(k, j);
            }
            setC(i, j, sum);
        }
    }
}

/**
 * @brief Initialize matrix with magic values for verification
 *
 * Each element is initialized with a unique value based on its logical position:
 * value = 10 + row + col*0.1
 * This allows easy verification of correctness while avoiding overflow in float16.
 *
 * @tparam T Data type
 * @param h_data Host pointer to matrix
 * @param rows Number of logical rows
 * @param cols Number of logical columns
 * @param storage_format 'R' for row-major, 'C' for column-major
 */
template<typename T>
void initMagicMatrix(T* h_data, int rows, int cols, char storage_format = 'R') {
    for (int i = 0; i < rows * cols; ++i) {
        T value = T(0.0f + i);
        h_data[i] = value;
    }
}

/**
 * @brief Print matrix for verification
 *
 * @tparam T Data type
 * @param h_data Host pointer to matrix
 * @param rows Number of logical rows
 * @param cols Number of logical columns
 * @param storage_format 'R' for row-major, 'C' for column-major
 * @param name Matrix name for printing
 */
template<typename T>
void printMatrix(const T* h_data, int rows, int cols, char storage_format = 'R', const char* name = "Matrix") {
    printf("\n%s (%d x %d, %s):\n", name, rows, cols,
           storage_format == 'R' ? "row-major" : "col-major");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = (storage_format == 'R') ? (i * cols + j) : (j * rows + i);
            printf("%8.2f ", (float)h_data[idx]);
        }
        printf("\n");
    }
}

} // namespace utils
} // namespace kebab

