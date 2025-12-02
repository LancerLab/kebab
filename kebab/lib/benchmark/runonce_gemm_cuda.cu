/**
 * @file runonce_gemm_cuda.cu
 * @brief Single-run CUDA baseline GEMM kernel for profiling
 *
 * This binary runs the CUDA baseline GEMM kernel exactly once (no loops, no warmup).
 * Designed for use with Nsight Compute profiling.
 *
 * Unlike runonce_gemm.cu which selects impl from config, this always runs the
 * CUDA baseline implementation (baseline::gemm).
 */
#include "kebab/utils/benchmark.h"
#include "kebab/cuda/cuda_gemm.h"
#include "kebab/utils/matrix_init.h"
#include "kebab/utils/cublas_reference.h"
#include "kebab/config/config_parser.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <random>
#include <iomanip>
#include <cassert>

using namespace kebab::benchmark;
using namespace kebab::utils;
using namespace kebab::config;

template<typename T>
void runOnceGEMMCuda(const ConfigParser& config) {
    std::string type_name;
    if constexpr (std::is_same_v<T, float>) {
        type_name = "float";
    } else if constexpr (std::is_same_v<T, __half>) {
        type_name = "half";
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        type_name = "bfloat16";
    }
    std::cout << "\n========================================" << std::endl;
    std::cout << "Single-Run CUDA Baseline GEMM (" << type_name << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    int version = config.getOperatorVersion("gemm");
    std::string init_method = config.getOperatorInitMethod("gemm");
    std::vector<std::string> opmode_list = config.getOperatorModes("gemm");
    auto matrix_sizes = config.getOperatorMatrixSizes("gemm");
    bool verbose = config.getOperatorVerbose("gemm");

    auto [init_A, init_B] = parseBinaryInitMethod(init_method);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Implementation: cuda (baseline) version " << version << std::endl;
    std::cout << "  Init method: " << init_method << std::endl;
    std::cout << "  Matrix size: " << matrix_sizes[0] << std::endl;
    std::cout << "  Mode: " << opmode_list[0] << std::endl;
    std::cout << "  Precision: " << type_name << std::endl;
    std::cout << "  Note: Running kernel ONCE (no loops, no warmup)" << std::endl;
    std::cout << std::endl;

    const auto& opmode = opmode_list[0];
    int size = matrix_sizes[0];
    int M = size, N = size, K = size;

    assert(opmode.length() >= 2);
    char lhs_format = opmode[0];
    char rhs_format = opmode[1];

    int A_storage_rows = (lhs_format == 'R') ? M : K;
    int A_storage_cols = (lhs_format == 'R') ? K : M;
    int B_storage_rows = (rhs_format == 'R') ? K : N;
    int B_storage_cols = (rhs_format == 'R') ? N : K;

    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Storage format: A=" << lhs_format << ", B=" << rhs_format << std::endl;

    std::vector<T> h_A(A_storage_rows * A_storage_cols);
    std::vector<T> h_B(B_storage_rows * B_storage_cols);
    std::vector<T> h_C(M * N);

    std::mt19937 gen(42);
    initializeMatrix(h_A.data(), M, K, init_A, gen, lhs_format);
    initializeMatrix(h_B.data(), K, N, init_B, gen, rhs_format);
    std::fill(h_C.begin(), h_C.end(), T{0});

    T *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_storage_rows * A_storage_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_B, B_storage_rows * B_storage_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_storage_rows * A_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_storage_rows * B_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));

    // Run CUDA baseline kernel once
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "\nRunning CUDA baseline kernel (version " << version << ") once..." << std::endl;

    baseline::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel execution complete!" << std::endl;

    // Verification using shared cuBLAS reference (same as bench_gemm)
    std::cout << "\nVerifying result against cuBLAS reference..." << std::endl;
    float tolerance = getVerificationTolerance<T>(size);
    bool passed = verifyCublasGemm(d_A, d_B, d_C, M, N, K, lhs_format, rhs_format, tolerance, true);

    std::cout << "\nReady for profiling analysis." << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Single-Run CUDA Baseline GEMM Profiling" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        auto& config = ConfigParser::getInstance("config.yaml");

        if (!config.isOperatorEnabled("gemm")) {
            std::cerr << "ERROR: GEMM operator is not enabled in config.yaml" << std::endl;
            return 1;
        }

        int gpu_id = config.getOperatorGpuId("gemm");
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (gpu_id >= 0 && gpu_id < device_count) {
            CUDA_CHECK(cudaSetDevice(gpu_id));
        } else if (gpu_id >= device_count) {
            std::cerr << "Warning: GPU " << gpu_id << " not available. Using GPU 0." << std::endl;
            CUDA_CHECK(cudaSetDevice(0));
            gpu_id = 0;
        } else {
            CUDA_CHECK(cudaGetDevice(&gpu_id));
        }

        std::cout << "Configuration:" << std::endl;
        std::cout << "  GPU ID: " << gpu_id << std::endl;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
        std::cout << "\nGPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;

        bool has_tensor_cores = (prop.major >= 7);
        auto precisions = config.getOperatorPrecisions("gemm");

        for (const auto& precision : precisions) {
            if (precision == "float32" || precision == "float") {
                runOnceGEMMCuda<float>(config);
            } else if (precision == "float16" || precision == "half") {
                if (has_tensor_cores) {
                    runOnceGEMMCuda<__half>(config);
                } else {
                    std::cout << "\nSkipping half (requires Tensor Cores)" << std::endl;
                }
            } else if (precision == "bfloat16" || precision == "bf16") {
                if (has_tensor_cores) {
                    runOnceGEMMCuda<__nv_bfloat16>(config);
                } else {
                    std::cout << "\nSkipping bfloat16 (requires Tensor Cores)" << std::endl;
                }
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

