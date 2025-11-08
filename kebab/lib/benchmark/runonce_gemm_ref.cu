#include "kebab/utils/benchmark.h"
#include "kebab/utils/matrix_init.h"
#include "kebab/utils/reference_gemm.h"
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
 * @brief Single-run cuBLAS reference kernel for profiling
 * 
 * This binary runs the cuBLAS GEMM kernel exactly once (no loops, no warmup).
 * Designed for use with Nsight Compute profiling to compare against custom kernels.
 * 
 * Usage: ./runonce_gemm_ref
 * 
 * Configuration is read from config.yaml
 */
template<typename T>
void runOnceGEMMRef(const ConfigParser& config) {
    std::string type_name = std::is_same_v<T, float> ? "float" : "half";
    std::cout << "\n========================================" << std::endl;
    std::cout << "Single-Run cuBLAS Reference Kernel (" << type_name << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get configuration
    std::string init_method = config.getOperatorInitMethod("gemm");
    std::vector<std::string> opmode_list = config.getOperatorModes("gemm");
    auto matrix_sizes = config.getOperatorMatrixSizes("gemm");
    bool verbose = config.getOperatorVerbose("gemm");
    
    // Parse init method
    auto [init_A, init_B] = parseBinaryInitMethod(init_method);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Implementation: cuBLAS (reference)" << std::endl;
    std::cout << "  Init method: " << init_method << std::endl;
    std::cout << "  Matrix size: " << matrix_sizes[0] << std::endl;
    std::cout << "  Mode: " << opmode_list[0] << std::endl;
    std::cout << "  Precision: " << type_name << std::endl;
    std::cout << "  Note: Running kernel ONCE (no loops, no warmup)" << std::endl;
    std::cout << std::endl;
    
    // Use first matrix size and first opmode
    const auto& opmode = opmode_list[0];
    int size = matrix_sizes[0];
    int M = size, N = size, K = size;
    
    // Parse opmode to determine storage formats
    assert(opmode.length() >= 2);
    char lhs_format = opmode[0];
    char rhs_format = opmode[1];
    
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
    // For small matrices, use magic values for verification
    if (M <= 4 && N <= 4 && K <= 4) {
        std::cout << "Using magic values for verification" << std::endl;
        initMagicMatrix(h_A.data(), M, K, lhs_format);
        initMagicMatrix(h_B.data(), K, N, rhs_format);
        printf("Input matrices:\n");
        printMatrix(h_A.data(), M, K, lhs_format, "A");
        printMatrix(h_B.data(), K, N, rhs_format, "B");
    } else {
        std::mt19937 gen(42);
        initializeMatrix(h_A.data(), M, K, init_A, gen, lhs_format);
        initializeMatrix(h_B.data(), K, N, init_B, gen, rhs_format);
    }
    std::fill(h_C.begin(), h_C.end(), T{0});
    
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_storage_rows * A_storage_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, B_storage_rows * B_storage_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_storage_rows * A_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_storage_rows * B_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
    
    // Run C++ reference GEMM for verification (only for small matrices)
    std::vector<T> h_C_cpp_ref(M * N, T{0});
    if (M <= 4 && N <= 4 && K <= 4) {
        std::cout << "\nRunning C++ reference GEMM for verification..." << std::endl;
        std::vector<T> h_A_verify(A_storage_rows * A_storage_cols);
        std::vector<T> h_B_verify(B_storage_rows * B_storage_cols);

        CUDA_CHECK(cudaMemcpy(h_A_verify.data(), d_A, A_storage_rows * A_storage_cols * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_B_verify.data(), d_B, B_storage_rows * B_storage_cols * sizeof(T), cudaMemcpyDeviceToHost));

        referenceGemmCpp<T>(h_A_verify.data(), h_B_verify.data(), h_C_cpp_ref.data(),
                            M, N, K, lhs_format, rhs_format, true);

        std::cout << "\nC++ Reference Result (col-major storage):" << std::endl;
        printMatrix(h_C_cpp_ref.data(), M, N, 'C', "C_cpp_ref");
    } else {
        std::cout << "\nSkipping C++ reference verification (matrix too large: " << M << "x" << N << "x" << K << ")" << std::endl;
    }

    // Synchronize before profiling
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "\nRunning cuBLAS reference kernel once..." << std::endl;

    // Get cuBLAS configuration
    auto config_cublas = getCublasGemmConfig(lhs_format, rhs_format, M, N, K);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Run cuBLAS kernel exactly once (no loops, no warmup)
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, config_cublas.opA, config_cublas.opB, M, N, K,
                    &alpha, d_A, config_cublas.ldA, d_B, config_cublas.ldB, &beta, d_C, config_cublas.ldC);
    } else {
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle, config_cublas.opA, config_cublas.opB, M, N, K,
                    &alpha, d_A, config_cublas.ldA, d_B, config_cublas.ldB, &beta, d_C, config_cublas.ldC);
    }

    // Synchronize after kernel
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "cuBLAS reference kernel execution complete!" << std::endl;

    // Verify cuBLAS result against C++ reference (only for small matrices)
    if (M <= 4 && N <= 4 && K <= 4) {
        std::cout << "\nVerifying cuBLAS result against C++ reference..." << std::endl;
        std::vector<T> h_C_cublas(M * N);
        CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));

        std::cout << "\ncuBLAS Result (col-major storage):" << std::endl;
        printMatrix(h_C_cublas.data(), M, N, 'C', "C_cublas");

        // Compare results (allow small tolerance for float16 precision)
        bool all_match = true;
        float max_diff = 0.0f;
        float tolerance = (std::is_same_v<T, __half>) ? 1.0f : 1e-5f;  // Higher tolerance for float16

        for (int i = 0; i < M * N; ++i) {
            float diff = std::abs((float)h_C_cublas[i] - (float)h_C_cpp_ref[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) {
                all_match = false;
                printf("Mismatch at index %d: cuBLAS=%.2f, C++_ref=%.2f, diff=%.2f\n",
                       i, (float)h_C_cublas[i], (float)h_C_cpp_ref[i], diff);
            }
        }

        printf("\nVerification Result: %s (max_diff=%.6f, tolerance=%.6f)\n",
               all_match ? "PASS" : "FAIL", max_diff, tolerance);
    }

    std::cout << "Ready for profiling analysis." << std::endl;

    // Cleanup
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Single-Run cuBLAS Reference Profiling Binary" << std::endl;
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
                runOnceGEMMRef<float>(config);
            } else if (precision == "float16" || precision == "half") {
                if (has_tensor_cores) {
                    runOnceGEMMRef<__half>(config);
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

