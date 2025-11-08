#include "kebab/utils/benchmark.h"
#include "kebab/cute/gemm.h"
#include "kebab/utils/matrix_init.h"
#include "kebab/utils/reference_gemm.h"
#include "kebab/cuda/cuda_gemm.h"
#include "kebab/config/config_parser.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include <iomanip>
#include <cassert>

using namespace kebab::benchmark;
using namespace kebab::utils;
using namespace kebab::config;

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

inline CublasGemmConfig getCublasGemmConfig(char lhs_format, char rhs_format, int M, int N, int K) {
    CublasGemmConfig config;
    config.ldC = M;

    if (lhs_format == 'R' && rhs_format == 'R') {
        config.opA = CUBLAS_OP_T;
        config.opB = CUBLAS_OP_T;
        config.ldA = K;
        config.ldB = N;
    } else if (lhs_format == 'R' && rhs_format == 'C') {
        config.opA = CUBLAS_OP_T;
        config.opB = CUBLAS_OP_N;
        config.ldA = K;
        config.ldB = K;
    } else if (lhs_format == 'C' && rhs_format == 'R') {
        config.opA = CUBLAS_OP_N;
        config.opB = CUBLAS_OP_T;
        config.ldA = M;
        config.ldB = N;
    } else {  // CC
        config.opA = CUBLAS_OP_N;
        config.opB = CUBLAS_OP_N;
        config.ldA = M;
        config.ldB = K;
    }

    return config;
}

/**
 * @brief Single-run GEMM kernel for profiling
 *
 * This binary runs the GEMM kernel exactly once (no loops, no warmup).
 * Designed for use with Nsight Compute profiling.
 *
 * Usage: ./runonce_gemm
 *
 * Configuration is read from config.yaml
 */
template<typename T>
void runOnceGEMM(const ConfigParser& config) {
    std::string type_name = std::is_same_v<T, float> ? "float" : "half";
    std::cout << "\n========================================" << std::endl;
    std::cout << "Single-Run GEMM Kernel (" << type_name << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get configuration
    std::string impl = config.getOperatorImpl("gemm");
    int version = config.getOperatorVersion("gemm");
    std::string init_method = config.getOperatorInitMethod("gemm");
    std::vector<std::string> opmode_list = config.getOperatorModes("gemm");
    auto matrix_sizes = config.getOperatorMatrixSizes("gemm");
    bool verbose = config.getOperatorVerbose("gemm");
    
    // Parse init method
    auto [init_A, init_B] = parseBinaryInitMethod(init_method);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Implementation: " << impl << " (version " << version << ")" << std::endl;
    std::cout << "  Init method: " << init_method << std::endl;
    std::cout << "  Matrix size: " << matrix_sizes[0] << std::endl;
    std::cout << "  Mode: " << opmode_list[0] << std::endl;
    std::cout << "  Precision: " << type_name << std::endl;
    std::cout << "  Verbose: " << (verbose ? "true" : "false") << std::endl;
    std::cout << "  Note: Running kernel ONCE (no loops, no warmup)" << std::endl;
    std::cout << std::endl;
    
    // Use first matrix size and first opmode
    // M = N = K = 4096
    // opmode = CR
    const auto& opmode = opmode_list[0];
    int size = matrix_sizes[0];
    int M = size, N = size, K = size;
    
    // Parse opmode to determine storage formats
    assert(opmode.length() >= 2);
    char lhs_format = opmode[0];
    char rhs_format = opmode[1];
    // lhs_format = C
    // rhs_format = R
    
    int A_storage_rows = (lhs_format == 'R') ? M : K;
    int A_storage_cols = (lhs_format == 'R') ? K : M;
    int B_storage_rows = (rhs_format == 'R') ? K : N;
    int B_storage_cols = (rhs_format == 'R') ? N : K;
    
    std::cout << "Matrix Configuration:" << std::endl;
    std::cout << "  M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "  A: " << M << "x" << K << " logical, stored as " 
              << A_storage_rows << "x" << A_storage_cols << " (" 
              << (lhs_format == 'R' ? "row-major" : "col-major") << ")" << std::endl;
    std::cout << "  B: " << K << "x" << N << " logical, stored as " 
              << B_storage_rows << "x" << B_storage_cols << " (" 
              << (rhs_format == 'R' ? "row-major" : "col-major") << ")" << std::endl;
    std::cout << std::endl;
    
    // Allocate host memory
    std::vector<T> h_A(A_storage_rows * A_storage_cols);
    std::vector<T> h_B(B_storage_rows * B_storage_cols);
    std::vector<T> h_C(M * N);

    // Initialize matrices
    std::mt19937 gen(42);
    initializeMatrix(h_A.data(), M, K, init_A, gen, lhs_format);
    initializeMatrix(h_B.data(), K, N, init_B, gen, rhs_format);
    std::fill(h_C.begin(), h_C.end(), T{0});

    // Print input matrices for small sizes
    if (verbose && M <= 128 && N <= 128 && K <= 128) {
        std::cout << "\nInput matrices:" << std::endl;
        printMatrix(h_A.data(), M, K, lhs_format, "A");
        printMatrix(h_B.data(), K, N, rhs_format, "B");
    }
    
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_storage_rows * A_storage_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, B_storage_rows * B_storage_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_storage_rows * A_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_storage_rows * B_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
    
    // Run cuBLAS reference for verification (only for matrices <= 128)
    std::vector<T> h_C_ref;
    bool run_verification = true;

    if (run_verification) {
        std::cout << "\nRunning cuBLAS reference for verification..." << std::endl;
        h_C_ref.resize(M * N, T{0});

        // Allocate device memory for reference result
        T *d_C_ref;
        CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_C_ref, 0, M * N * sizeof(T)));

        // Get cuBLAS configuration
        auto config_cublas = getCublasGemmConfig(lhs_format, rhs_format, M, N, K);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Run cuBLAS kernel
        if constexpr (std::is_same_v<T, float>) {
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle, config_cublas.opA, config_cublas.opB, M, N, K,
                        &alpha, d_A, config_cublas.ldA, d_B, config_cublas.ldB,
                        &beta, d_C_ref, config_cublas.ldC);
        } else {
            const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            cublasHgemm(handle, config_cublas.opA, config_cublas.opB, M, N, K,
                        &alpha, d_A, config_cublas.ldA, d_B, config_cublas.ldB,
                        &beta, d_C_ref, config_cublas.ldC);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy reference result to host
        CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, M * N * sizeof(T), cudaMemcpyDeviceToHost));

        // Print cuBLAS result for small matrices
        if (verbose && M <= 128 && N <= 128) {
            std::cout << "\ncuBLAS Reference Result (col-major storage):" << std::endl;
            printMatrix(h_C_ref.data(), M, N, 'C', "C_cublas_ref");
        }

        // Cleanup cuBLAS
        cublasDestroy(handle);
        CUDA_CHECK(cudaFree(d_C_ref));

        std::cout << "cuBLAS reference complete." << std::endl;
    } else {
        std::cout << "\nSkipping verification (matrix size " << M << "x" << N << "x" << K
                  << " exceeds threshold of 128)" << std::endl;
    }

    // Synchronize before profiling
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\nRunning " << impl << " kernel once..." << std::endl;

    // Run kernel exactly once (no loops, no warmup)
    if (impl == "cute") {
        kebab::cute::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
    } else {
        baseline::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
    }

    // Synchronize after kernel
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Kernel execution complete!" << std::endl;

    // Verify result against cuBLAS reference
    if (run_verification) {
        std::cout << "\nVerifying result against cuBLAS reference..." << std::endl;
        std::vector<T> h_C_result(M * N);
        CUDA_CHECK(cudaMemcpy(h_C_result.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

        // Print kernel result for small matrices
        if (verbose && M <= 128 && N <= 128) {
            std::cout << "\n" << impl << " Kernel Result (col-major storage):" << std::endl;
            printMatrix(h_C_result.data(), M, N, 'C', "C_kernel");
        }

        // Compare results
        bool all_match = true;
        float max_diff = 0.0f;
        float tolerance = (std::is_same_v<T, __half>) ? 1e-2f : 1e-5f;  // Tolerance for float16
        int mismatch_count = 0;
        const int max_print = 10;  // Only print first 10 mismatches

        for (int i = 0; i < M * N; ++i) {
            float diff = std::abs((float)h_C_result[i] - (float)h_C_ref[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) {
                all_match = false;
                if (mismatch_count < max_print) {
                    printf("Mismatch at index %d: kernel=%.6f, cuBLAS=%.6f, diff=%.6f\n",
                           i, (float)h_C_result[i], (float)h_C_ref[i], diff);
                }
                mismatch_count++;
            }
        }

        if (mismatch_count > max_print) {
            printf("... and %d more mismatches (total %d/%d elements)\n",
                   mismatch_count - max_print, mismatch_count, M * N);
        }

        printf("\nVerification Result: %s\n", all_match ? "PASS ✓" : "FAIL ✗");
        printf("  Max difference: %.6f\n", max_diff);
        printf("  Tolerance: %.6f\n", tolerance);
        printf("  Mismatches: %d / %d (%.2f%%)\n",
               mismatch_count, M * N, 100.0f * mismatch_count / (M * N));
    }

    std::cout << "\nReady for profiling analysis." << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Single-Run GEMM Profiling Binary" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // Load configuration (singleton)
        auto& config = ConfigParser::getInstance("config.yaml");

        // Check if GEMM is enabled
        if (!config.isOperatorEnabled("gemm")) {
            std::cerr << "ERROR: GEMM operator is not enabled in config.yaml" << std::endl;
            return 1;
        }

        // Set GPU device
        int gpu_id = config.getOperatorGpuId("gemm");
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (gpu_id >= 0 && gpu_id < device_count) {
            CUDA_CHECK(cudaSetDevice(gpu_id));
        } else if (gpu_id >= device_count) {
            std::cerr << "Warning: GPU " << gpu_id << " not available (only " << device_count
                      << " GPUs detected). Using GPU 0." << std::endl;
            CUDA_CHECK(cudaSetDevice(0));
            gpu_id = 0;
        } else {
            CUDA_CHECK(cudaGetDevice(&gpu_id));
        }

        std::cout << "Configuration:" << std::endl;
        std::cout << "  GPU ID: " << gpu_id << std::endl;

        // Get GPU properties
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        std::cout << "\nGPU Information:" << std::endl;
        std::cout << "  Device: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;

        bool has_tensor_cores = (prop.major >= 7);
        std::cout << "  Tensor Cores: " << (has_tensor_cores ? "Supported" : "Not Supported") << std::endl;

        // Get precisions from config
        auto precisions = config.getOperatorPrecisions("gemm");

        // Run for each precision
        for (const auto& precision : precisions) {
            if (precision == "float32" || precision == "float") {
                runOnceGEMM<float>(config);
            } else if (precision == "float16" || precision == "half") {
                if (has_tensor_cores) {
                    runOnceGEMM<__half>(config);
                } else {
                    std::cout << "\nSkipping half precision (requires Tensor Core support)" << std::endl;
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

