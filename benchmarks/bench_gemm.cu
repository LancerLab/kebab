#include "benchmark_runner.h"
#include "cutekernellib/operators/gemm.h"
#include "cutekernellib/utils/matrix_init.h"
#include "cutekernellib/utils/matrix_print.h"
#include "cuda_gemm.h"
#include "cutekernellib/config/config_parser.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include <iomanip>

using namespace cutekernellib::benchmark;
using namespace cutekernellib::utils;
using namespace cutekernellib::config;

/**
 * @brief Configure cuBLAS parameters based on storage format
 */
struct CublasConfig {
    cublasOperation_t opA;
    cublasOperation_t opB;
    int ldA;
    int ldB;
    int ldC;
};

inline CublasConfig getCublasConfig(const std::string& opmode, int M, int N, int K) {
    char lhs_format = (opmode.length() >= 1) ? opmode[0] : 'R';
    char rhs_format = (opmode.length() >= 2) ? opmode[1] : 'R';
    
    CublasConfig config;
    config.ldC = M;  // C is always row-major output (M×N)
    
    if (lhs_format == 'R' && rhs_format == 'R') {
        // RR: Both row-major
        // A is M×K row-major (ldA=K), B is K×N row-major (ldB=N)
        // cuBLAS computes: C^T = B^T × A^T
        config.opA = CUBLAS_OP_T;  // A^T
        config.opB = CUBLAS_OP_T;  // B^T
        config.ldA = K;
        config.ldB = N;
    } else if (lhs_format == 'R' && rhs_format == 'C') {
        // RC: A row-major, B column-major
        // A is M×K row-major (ldA=K), B is K×N column-major (ldB=K)
        // cuBLAS computes: C^T = B × A^T (B is already transposed in storage)
        config.opA = CUBLAS_OP_T;  // A^T
        config.opB = CUBLAS_OP_N;  // B (no transpose needed)
        config.ldA = K;
        config.ldB = K;
    } else if (lhs_format == 'C' && rhs_format == 'R') {
        // CR: A column-major, B row-major
        // A is M×K column-major (ldA=M), B is K×N row-major (ldB=N)
        // cuBLAS computes: C^T = B^T × A (A is already transposed in storage)
        config.opA = CUBLAS_OP_N;  // A (no transpose needed)
        config.opB = CUBLAS_OP_T;  // B^T
        config.ldA = M;
        config.ldB = N;
    } else {  // CC
        // CC: Both column-major
        // A is M×K column-major (ldA=M), B is K×N column-major (ldB=K)
        // cuBLAS computes: C^T = B × A (both already transposed in storage)
        config.opA = CUBLAS_OP_N;  // A (no transpose needed)
        config.opB = CUBLAS_OP_N;  // B (no transpose needed)
        config.ldA = M;
        config.ldB = K;
    }
    
    return config;
}

/**
 * @brief Verify GEMM correctness against cuBLAS reference
 */
template<typename T>
bool verifyGEMM(const T* A, const T* B, const T* C_test, int M, int N, int K, 
                const std::string& opmode, float tolerance = 1e-3f) {
    // Allocate reference result
    std::vector<T> C_ref(M * N);
    T* d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_C_ref, 0, M * N * sizeof(T)));
    
    // Run cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Get cuBLAS configuration based on storage format
    auto config = getCublasConfig(opmode, M, N, K);
    
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, config.opB, config.opA, N, M, K,
                    &alpha, B, config.ldB, A, config.ldA, &beta, d_C_ref, config.ldC);
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle, config.opB, config.opA, N, M, K,
                    &alpha, B, config.ldB, A, config.ldA, &beta, d_C_ref, config.ldC);
    }
    
    cublasDestroy(handle);
    
    // Copy reference result to host
    CUDA_CHECK(cudaMemcpy(C_ref.data(), d_C_ref, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_C_ref));
    
    // Copy test result to host
    std::vector<T> C_test_host(M * N);
    CUDA_CHECK(cudaMemcpy(C_test_host.data(), C_test, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    
    // Compare results
    bool correct = true;
    int errors = 0;
    const int max_errors_to_show = 5;
    
    for (int i = 0; i < M * N && errors < max_errors_to_show; ++i) {
        float ref_val, test_val;
        
        if constexpr (std::is_same_v<T, float>) {
            ref_val = C_ref[i];
            test_val = C_test_host[i];
        } else {
            ref_val = __half2float(C_ref[i]);
            test_val = __half2float(C_test_host[i]);
        }
        
        float abs_error = std::abs(test_val - ref_val);
        float rel_error = (ref_val != 0.0f) ? abs_error / std::abs(ref_val) : abs_error;
        
        if (abs_error > tolerance && rel_error > tolerance) {
            if (errors == 0) {
                std::cout << "  Verification errors found:" << std::endl;
            }
            std::cout << "    Element " << i << ": expected " << ref_val 
                      << ", got " << test_val << " (error: " << abs_error << ")" << std::endl;
            errors++;
            correct = false;
        }
    }
    
    if (errors >= max_errors_to_show) {
        std::cout << "    ... (showing first " << max_errors_to_show << " errors)" << std::endl;
    }
    
    return correct;
}

/**
 * @brief Benchmark GEMM with configuration from config.yaml
 */
template<typename T>
void benchmarkGEMM(const ConfigParser& config) {
    std::string type_name = std::is_same_v<T, float> ? "float" : "half";
    std::cout << "\n========================================" << std::endl;
    std::cout << "GEMM Benchmark (" << type_name << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get configuration
    std::string impl = config.getOperatorImpl("gemm");
    int version = config.getOperatorVersion("gemm");
    std::string init_method = config.getOperatorInitMethod("gemm");
    std::vector<std::string> opmode_list = config.getOperatorModes("gemm");
    auto matrix_sizes = config.getOperatorMatrixSizes("gemm");
    int warmup_runs = config.getWarmupRuns();
    int measurement_runs = config.getMeasurementRuns();
    bool verbose = config.getOperatorVerbose("gemm");
    
    // Parse init method
    auto [init_A, init_B] = parseBinaryInitMethod(init_method);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Implementation: " << impl << " (version " << version << ")" << std::endl;
    std::cout << "  Modes: ";
    // verbose Mode in modelist
    for (const auto& opmode : opmode_list) {
        std::cout << opmode << ", ";
    }
    std::cout << "  Init method: " << init_method << std::endl;
    std::cout << "  Verbose mode: " << (verbose ? "enabled" : "disabled") << std::endl;
    std::cout << "  Matrix sizes: ";
    for (size_t i = 0; i < matrix_sizes.size(); ++i) {
        std::cout << matrix_sizes[i];
        if (i < matrix_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;
    
    BenchmarkRunner runner(warmup_runs, measurement_runs);
    std::vector<BenchmarkResult> results;
    
    // Create CSV writer
    std::string csv_filename = "bench_results/gemm_results_" + type_name + "_" + impl + ".csv";
    CSVWriter csv(csv_filename);
    if (!csv.isOpen()) {
        std::cerr << "ERROR: Failed to create CSV file: " << csv_filename << std::endl;
        return;
    }
    csv.writeHeader();
    
    // Print benchmark header
    BenchmarkRunner::printHeader();
    
    // Random number generator for initialization
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    for (const auto& opmode : opmode_list) {
        for (int size : matrix_sizes) {
            int M = size, N = size, K = size; // Square matrices
            
            // Parse opmode to determine storage formats
            char lhs_format = (opmode.length() >= 1) ? opmode[0] : 'R';  // R or C
            char rhs_format = (opmode.length() >= 2) ? opmode[1] : 'R';  // R or C
            
            // Logical dimensions for GEMM: C(M×N) = A(M×K) × B(K×N)
            // Storage dimensions depend on format:
            // - Row-major A(M×K): stored as M×K
            // - Column-major A(M×K): stored as K×M (transposed in memory)
            int A_storage_rows = (lhs_format == 'R') ? M : K;
            int A_storage_cols = (lhs_format == 'R') ? K : M;
            int B_storage_rows = (rhs_format == 'R') ? K : N;
            int B_storage_cols = (rhs_format == 'R') ? N : K;
            
            std::cout << "\nTesting matrix size: " << M << "x" << N << "x" << K 
                      << " (mode: " << opmode << ")" << std::endl;
            std::cout << "  A: " << M << "x" << K << " logical, stored as " 
                      << A_storage_rows << "x" << A_storage_cols << " (" 
                      << (lhs_format == 'R' ? "row-major" : "col-major") << ")" << std::endl;
            std::cout << "  B: " << K << "x" << N << " logical, stored as " 
                      << B_storage_rows << "x" << B_storage_cols << " (" 
                      << (rhs_format == 'R' ? "row-major" : "col-major") << ")" << std::endl;
            std::cout << "  Init: A=" << init_method.substr(0, init_method.find('-')) 
                      << ", B=" << init_method.substr(init_method.find('-') + 1) << std::endl;
            
            // Allocate host memory (using storage dimensions)
            std::vector<T> h_A(A_storage_rows * A_storage_cols);
            std::vector<T> h_B(B_storage_rows * B_storage_cols);
            std::vector<T> h_C(M * N);
            
            // Initialize matrices with logical dimensions and storage format
            // This ensures the initialization pattern is based on logical matrix structure
            initializeMatrix(h_A.data(), M, K, init_A, gen, lhs_format);
            initializeMatrix(h_B.data(), K, N, init_B, gen, rhs_format);
            std::fill(h_C.begin(), h_C.end(), T{0});
            
            // Print initialized matrices if verbose mode is enabled
            if (verbose) {
                printMatrix(h_A, M, K, "A", lhs_format, 16);
                printMatrix(h_B, K, N, "B", rhs_format, 16);
            }
            
            // Allocate device memory
            T *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, A_storage_rows * A_storage_cols * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_B, B_storage_rows * B_storage_cols * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_storage_rows * A_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_storage_rows * B_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
            
            // Benchmark selected implementation
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
            
            auto kernel = [&]() {
                if (impl == "cute") {
                    cutekernellib::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
                } else {
                    baseline::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
                }
            };
            float latency = runner.measureLatency(kernel);
            
            // Calculate GFLOPS
            size_t flops = 2ULL * M * N * K;
            float gflops = runner.calculateGFLOPS(flops, latency);
            
            // Benchmark cuBLAS for comparison (using same configuration as verification)
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
            auto config = getCublasConfig(opmode, M, N, K);
            auto cublas_kernel = [&]() {
                cublasHandle_t handle;
                cublasCreate(&handle);
                if constexpr (std::is_same_v<T, float>) {
                    const float alpha = 1.0f, beta = 0.0f;
                    cublasSgemm(handle, config.opB, config.opA, N, M, K,
                                &alpha, d_B, config.ldB, d_A, config.ldA, &beta, d_C, config.ldC);
                } else {
                    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                    cublasHgemm(handle, config.opB, config.opA, N, M, K,
                                &alpha, d_B, config.ldB, d_A, config.ldA, &beta, d_C, config.ldC);
                }
                cublasDestroy(handle);
            };
            
            float cublas_latency = runner.measureLatency(cublas_kernel);
            float cublas_gflops = runner.calculateGFLOPS(flops, cublas_latency);
            
            // Print performance results
            float speedup = cublas_gflops > 0 ? gflops / cublas_gflops : 0.0f;
            
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "GEMM                " << std::setw(12) << impl 
                      << std::setw(12) << size
                      << std::setw(16) << latency
                      << std::setw(24) << gflops
                      << std::setw(14) << speedup << "     x" << std::endl;
        
            std::cout << "GEMM                " << std::setw(12) << "cuBLAS"
                      << std::setw(12) << size
                      << std::setw(16) << cublas_latency
                      << std::setw(24) << cublas_gflops
                      << std::setw(14) << "1.000" << "     x" << std::endl;
            
            std::cout << std::endl;  // Add blank line after performance table
            
            // Verify correctness (after performance results)
            std::cout << "  Verifying " << impl << " implementation... ";
            float tolerance;
            if constexpr (std::is_same_v<T, __half>) {
                tolerance = (size <= 512) ? 0.15f : (size <= 1024) ? 0.25f : 1.5f;
            } else {
                tolerance = 1e-3f;
            }
        
            // Reset C for verification
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
            if (impl == "cute") {
                cutekernellib::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
            } else {
                baseline::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
            }
            
            bool correct = verifyGEMM(d_A, d_B, d_C, M, N, K, opmode, tolerance);
        std::cout << (correct ? "✓ PASSED" : "✗ FAILED") << std::endl;
        
        // Verbose output: show detailed matrix comparison
        if (verbose) {
            // Copy matrices to host for printing
            std::vector<T> h_A_copy(M * K), h_B_copy(K * N), h_C_result(M * N), h_C_reference(M * N);
            CUDA_CHECK(cudaMemcpy(h_A_copy.data(), d_A, M * K * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_B_copy.data(), d_B, K * N * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_C_result.data(), d_C, M * N * sizeof(T), cudaMemcpyDeviceToHost));
            
            // Get reference result from cuBLAS (using same configuration as verification)
            T* d_C_ref;
            CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(T)));
            CUDA_CHECK(cudaMemset(d_C_ref, 0, M * N * sizeof(T)));
            
            auto config = getCublasConfig(opmode, M, N, K);
            cublasHandle_t handle;
            cublasCreate(&handle);
            if constexpr (std::is_same_v<T, float>) {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(handle, config.opB, config.opA, N, M, K,
                           &alpha, d_B, config.ldB, d_A, config.ldA, &beta, d_C_ref, config.ldC);
            } else {
                const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                cublasHgemm(handle, config.opB, config.opA, N, M, K,
                           &alpha, d_B, config.ldB, d_A, config.ldA, &beta, d_C_ref, config.ldC);
            }
            cublasDestroy(handle);
            
            CUDA_CHECK(cudaMemcpy(h_C_reference.data(), d_C_ref, M * N * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_C_ref));
            
            // Print individual matrices for inspection
            std::cout << "\n" << Colors::BOLD << Colors::BLUE 
                      << "═══════════════════════════════════════════════════════════════" 
                      << Colors::RESET << std::endl;
            std::cout << Colors::BOLD << Colors::BLUE << "  Matrix Verification Details" 
                      << Colors::RESET << std::endl;
            std::cout << Colors::BOLD << Colors::BLUE 
                      << "═══════════════════════════════════════════════════════════════" 
                      << Colors::RESET << std::endl;
            
            int max_display = std::min(16, std::min(M, N));
            
            // Print input matrices
            printMatrix(h_A_copy, M, K, "A (Input)", lhs_format, max_display);
            printMatrix(h_B_copy, K, N, "B (Input)", rhs_format, max_display);
            
            // Print output matrices
            printMatrix(h_C_result, M, N, "C (CuTe Result)", 'R', max_display);
            printMatrix(h_C_reference, M, N, "C (cuBLAS Reference)", 'R', max_display);
            
            if (!correct) {
                // Configure matrix printing for comparison
                MatrixPrintConfig print_config;
                print_config.max_rows = std::min(16, M);
                print_config.max_cols = std::min(16, N);
                print_config.precision = 3;
                print_config.width = 8;
                print_config.error_threshold = tolerance;
                print_config.use_colors = true;
                print_config.show_comparison = true;
                print_config.comparison_cols = std::min(8, N);  // Limit columns for readability
                
                // Print result comparison
                std::string comparison_title = impl + " vs cuBLAS (Matrix C)";
                printMatrixComparison(h_C_result, h_C_reference, M, N, comparison_title, print_config);
            }
        }
        
        std::cout << "  Performance Analysis:" << std::endl;
        std::cout << "    " << impl << " vs cuBLAS:     " 
                  << std::fixed << std::setprecision(1) 
                  << (speedup * 100.0f) << "% performance" << std::endl;
        
        // Save to CSV
        BenchmarkResult result;
        result.operator_name = "GEMM";
        result.variant = impl;
        result.batch_size = size;
        result.latency_ms = latency;
        result.throughput_gbps = gflops;  // Using GFLOPS field for throughput
        result.speedup_ratio = speedup;
        csv.writeResult(result);
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "GEMM Benchmark Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results saved to: " << csv_filename << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "CuTeKernelLib GEMM Benchmark" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Load configuration
    auto& config = ConfigParser::getInstance("config.yaml");
    std::cout << "Configuration loaded successfully from: config.yaml" << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Warmup runs:      " << config.getWarmupRuns() << std::endl;
    std::cout << "  Measurement runs: " << config.getMeasurementRuns() << std::endl;
    
    // Get matrix sizes from config
    auto matrix_sizes = config.getOperatorMatrixSizes("gemm");
    std::cout << "  Matrix sizes:     ";
    for (size_t i = 0; i < matrix_sizes.size(); ++i) {
        std::cout << matrix_sizes[i];
        if (i < matrix_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Check GPU capabilities
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\nGPU Information:" << std::endl;
    std::cout << "  Device:           " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Memory:           " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
    
    bool has_tensor_cores = (prop.major >= 7);
    std::cout << "  Tensor Cores:     " << (has_tensor_cores ? "Supported" : "Not Supported") << std::endl;
    
    // Get precisions from config
    auto precisions = config.getOperatorPrecisions("gemm");
    
    // Run benchmarks for each precision
    for (const auto& precision : precisions) {
        if (precision == "float32" || precision == "float") {
            benchmarkGEMM<float>(config);
        } else if (precision == "float16" || precision == "half") {
            if (has_tensor_cores) {
                benchmarkGEMM<__half>(config);
            } else {
                std::cout << "\nSkipping half precision benchmark (requires Tensor Core support)" << std::endl;
            }
        }
    }
    
    return 0;
}
