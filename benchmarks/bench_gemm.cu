#include "benchmark_runner.h"
#include "cutekernellib/operators/gemm.h"
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

/**
 * @brief Initialize matrix with random values
 */
template<typename T>
void initializeMatrix(T* matrix, int size, std::mt19937& gen) {
    if constexpr (std::is_same_v<T, float>) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < size; ++i) {
            matrix[i] = dist(gen);
        }
    } else if constexpr (std::is_same_v<T, __half>) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < size; ++i) {
            matrix[i] = __float2half(dist(gen));
        }
    }
}

/**
 * @brief Verify GEMM correctness against cuBLAS reference
 */
template<typename T>
bool verifyGEMM(const T* A, const T* B, const T* C_test, int M, int N, int K, float tolerance = 1e-3f) {
    // Allocate reference result
    std::vector<T> C_ref(M * N);
    T* d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_C_ref, 0, M * N * sizeof(T)));
    
    // Run cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, B, N, A, K, &beta, d_C_ref, N);
    } else if constexpr (std::is_same_v<T, __half>) {
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, B, N, A, K, &beta, d_C_ref, N);
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
 * @brief Benchmark GEMM implementations
 */
template<typename T>
void benchmarkGEMM(const std::vector<int>& matrix_sizes, int warmup_runs, int measurement_runs) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GEMM Benchmark (" << (std::is_same_v<T, float> ? "float" : "half") << ")" << std::endl;
    std::cout << "========================================" << std::endl;
    
    BenchmarkRunner runner(warmup_runs, measurement_runs);
    std::vector<BenchmarkResult> results;
    
    // Create CSV writer
    std::string csv_filename = "bench_results/gemm_results_" + 
                              std::string(std::is_same_v<T, float> ? "float" : "half") + ".csv";
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
    
    for (int size : matrix_sizes) {
        int M = size, N = size, K = size; // Square matrices
        
        std::cout << "\nTesting matrix size: " << M << "x" << N << "x" << K << std::endl;
        
        // Allocate host memory
        std::vector<T> h_A(M * K), h_B(K * N), h_C(M * N);
        
        // Initialize matrices with random data
        initializeMatrix(h_A.data(), M * K, gen);
        initializeMatrix(h_B.data(), K * N, gen);
        std::fill(h_C.begin(), h_C.end(), T{0});
        
        // Allocate device memory
        T *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));
        
        // Benchmark CuTe implementation
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
        auto cute_kernel = [&]() {
            cutekernellib::gemm(d_A, d_B, d_C, M, N, K);
        };
        
        float cute_latency = runner.measureLatency(cute_kernel);
        
        // Verify correctness (use relaxed tolerance for half precision)
        // Half precision accumulates more error with larger matrices
        std::cout << "  Verifying CuTe implementation... ";
        float tolerance;
        if constexpr (std::is_same_v<T, __half>) {
            // Scale tolerance with matrix size for half precision
            // Larger matrices accumulate more rounding errors (FP16 has only 10-bit mantissa)
            tolerance = (size <= 512) ? 0.15f : (size <= 1024) ? 0.25f : 1.5f;
        } else {
            tolerance = 1e-3f;
        }
        bool cute_correct = verifyGEMM(d_A, d_B, d_C, M, N, K, tolerance);
        std::cout << (cute_correct ? "✓ PASSED" : "✗ FAILED") << std::endl;
        
        // Calculate GFLOPS for CuTe (2*M*N*K operations)
        size_t flops = 2ULL * M * N * K;
        float cute_gflops = runner.calculateGFLOPS(flops, cute_latency);
        
        // Benchmark CUDA baseline
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
        auto cuda_kernel = [&]() {
            baseline::gemm(d_A, d_B, d_C, M, N, K);
        };
        
        float cuda_latency = runner.measureLatency(cuda_kernel);
        
        // Verify correctness (use relaxed tolerance for half precision)
        // Half precision accumulates more error with larger matrices
        std::cout << "  Verifying CUDA baseline... ";
        if constexpr (std::is_same_v<T, __half>) {
            // Scale tolerance with matrix size for half precision
            tolerance = (size <= 512) ? 0.15f : (size <= 1024) ? 0.25f : 1.5f;
        } else {
            tolerance = 1e-3f;
        }
        bool cuda_correct = verifyGEMM(d_A, d_B, d_C, M, N, K, tolerance);
        std::cout << (cuda_correct ? "✓ PASSED" : "✗ FAILED") << std::endl;
        
        // Calculate GFLOPS for CUDA baseline
        float cuda_gflops = runner.calculateGFLOPS(flops, cuda_latency);
        
        // Benchmark cuBLAS reference
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        auto cublas_kernel = [&]() {
            if constexpr (std::is_same_v<T, float>) {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                           &alpha, d_B, N, d_A, K, &beta, d_C, N);
            } else if constexpr (std::is_same_v<T, __half>) {
                const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                           &alpha, d_B, N, d_A, K, &beta, d_C, N);
            }
        };
        
        float cublas_latency = runner.measureLatency(cublas_kernel);
        float cublas_gflops = runner.calculateGFLOPS(flops, cublas_latency);
        
        cublasDestroy(handle);
        
        // Calculate speedups (relative to cuBLAS)
        float cute_speedup = cublas_latency / cute_latency;
        float cuda_speedup = cublas_latency / cuda_latency;
        
        // Store results (using GFLOPS as throughput metric)
        BenchmarkResult cute_result("GEMM", "CuTe", size, cute_latency, cute_gflops, cute_speedup);
        BenchmarkResult cuda_result("GEMM", "CUDA", size, cuda_latency, cuda_gflops, cuda_speedup);
        BenchmarkResult cublas_result("GEMM", "cuBLAS", size, cublas_latency, cublas_gflops, 1.0f);
        
        results.push_back(cute_result);
        results.push_back(cuda_result);
        results.push_back(cublas_result);
        
        // Print results
        BenchmarkRunner::printResult(cute_result);
        BenchmarkRunner::printResult(cuda_result);
        BenchmarkRunner::printResult(cublas_result);
        
        // Write to CSV
        csv.writeResult(cute_result);
        csv.writeResult(cuda_result);
        csv.writeResult(cublas_result);
        
        // Performance analysis
        std::cout << "  Performance Analysis:" << std::endl;
        std::cout << "    CuTe vs cuBLAS:     " << std::fixed << std::setprecision(1) 
                  << (cute_gflops / cublas_gflops * 100.0f) << "% performance" << std::endl;
        std::cout << "    CUDA vs cuBLAS:     " << std::fixed << std::setprecision(1) 
                  << (cuda_gflops / cublas_gflops * 100.0f) << "% performance" << std::endl;
        
        // Clean up device memory
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "GEMM Benchmark Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results saved to: " << csv_filename << std::endl;
    
    // Summary statistics
    if (!results.empty()) {
        float avg_cute_gflops = 0.0f, avg_cuda_gflops = 0.0f, avg_cublas_gflops = 0.0f;
        int count = 0;
        
        for (const auto& result : results) {
            if (result.variant == "CuTe") {
                avg_cute_gflops += result.throughput_gbps; // Actually GFLOPS
                count++;
            } else if (result.variant == "CUDA") {
                avg_cuda_gflops += result.throughput_gbps;
            } else if (result.variant == "cuBLAS") {
                avg_cublas_gflops += result.throughput_gbps;
            }
        }
        
        if (count > 0) {
            avg_cute_gflops /= count;
            avg_cuda_gflops /= count;
            avg_cublas_gflops /= count;
            
            std::cout << "\nAverage Performance:" << std::endl;
            std::cout << "  CuTe:    " << std::fixed << std::setprecision(1) << avg_cute_gflops << " GFLOPS" << std::endl;
            std::cout << "  CUDA:    " << std::fixed << std::setprecision(1) << avg_cuda_gflops << " GFLOPS" << std::endl;
            std::cout << "  cuBLAS:  " << std::fixed << std::setprecision(1) << avg_cublas_gflops << " GFLOPS" << std::endl;
            std::cout << "\nRelative Performance:" << std::endl;
            std::cout << "  CuTe vs cuBLAS:  " << std::fixed << std::setprecision(1) 
                      << (avg_cute_gflops / avg_cublas_gflops * 100.0f) << "%" << std::endl;
            std::cout << "  CUDA vs cuBLAS:  " << std::fixed << std::setprecision(1) 
                      << (avg_cuda_gflops / avg_cublas_gflops * 100.0f) << "%" << std::endl;
        }
    }
    
    std::cout << "========================================" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "CuTeKernelLib GEMM Benchmark" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        // Load configuration
        auto& config = cutekernellib::config::ConfigParser::getInstance("config.yaml");
        
        int warmup = config.getWarmupRuns();
        int measurement = config.getMeasurementRuns();
        
        // Get matrix sizes from config (use batch_sizes as matrix sizes)
        auto batch_sizes = config.getBatchSizes();
        std::vector<int> matrix_sizes;
        
        // For GEMM, we can use larger sizes than batch_sizes if specified in operators.gemm.matrix_sizes
        try {
            matrix_sizes = config.getOperatorMatrixSizes("gemm");
            if (matrix_sizes.empty()) {
                matrix_sizes = batch_sizes;
            }
        } catch (...) {
            // Fall back to batch_sizes if gemm.matrix_sizes not found
            matrix_sizes = batch_sizes;
        }
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Warmup runs:      " << warmup << std::endl;
        std::cout << "  Measurement runs: " << measurement << std::endl;
        std::cout << "  Matrix sizes:     ";
        for (size_t i = 0; i < matrix_sizes.size(); ++i) {
            std::cout << matrix_sizes[i];
            if (i < matrix_sizes.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Get GPU information
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        std::cout << "\nGPU Information:" << std::endl;
        std::cout << "  Device:           " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory:           " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
        
        // Check for Tensor Core support
        bool has_tensor_cores = (prop.major >= 7); // Volta and newer
        std::cout << "  Tensor Cores:     " << (has_tensor_cores ? "Supported" : "Not Supported") << std::endl;
        
        if (!has_tensor_cores) {
            std::cout << "\n⚠ WARNING: This GPU does not support Tensor Cores." << std::endl;
            std::cout << "  GEMM performance may be limited compared to Tensor Core enabled GPUs." << std::endl;
        }
        
        // Run benchmarks for float
        benchmarkGEMM<float>(matrix_sizes, warmup, measurement);
        
        // Run benchmarks for half precision if supported
        if (has_tensor_cores) {
            benchmarkGEMM<__half>(matrix_sizes, warmup, measurement);
        } else {
            std::cout << "\nSkipping half precision benchmark (requires Tensor Core support)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}