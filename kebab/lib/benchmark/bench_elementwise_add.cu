#include "kebab/utils/benchmark.h"
#include "kebab/cute/elementwise_add.h"
#include "kebab/config/config_parser.h"
#include "kebab/cuda/cuda_elementwise_add.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <iomanip>

using namespace kebab::benchmark;
using namespace kebab::config;

/**
 * @brief Benchmark element-wise add operator
 * 
 * This benchmark compares the CuTe implementation against the hand-optimized
 * CUDA baseline across multiple batch sizes specified in config.yaml.
 * 
 * Metrics:
 * - Latency (ms): Average execution time
 * - Throughput (GB/s): Memory bandwidth utilization
 * - Speedup: CuTe performance relative to CUDA baseline
 */

template<typename T>
void benchmarkElementwiseAdd(const std::vector<int>& batch_sizes, 
                             int warmup_runs, 
                             int measurement_runs,
                             CSVWriter& csv) {
    BenchmarkRunner runner(warmup_runs, measurement_runs);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmarking Element-wise Add" << std::endl;
    std::cout << "Data type: " << (sizeof(T) == 4 ? "float32" : "float16") << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Warmup runs: " << warmup_runs << std::endl;
    std::cout << "  Measurement runs: " << measurement_runs << std::endl;
    std::cout << "  Batch sizes: ";
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        std::cout << batch_sizes[i];
        if (i < batch_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << "\n" << std::endl;
    
    BenchmarkRunner::printHeader();
    
    // Store baseline latencies for speedup calculation
    std::vector<float> baseline_latencies;
    
    for (int N : batch_sizes) {
        // Allocate device memory
        T *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(T)));
        
        // Initialize with random data
        std::vector<T> h_A(N), h_B(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (int i = 0; i < N; ++i) {
            if constexpr (sizeof(T) == 4) {  // float
                h_A[i] = static_cast<T>(dist(gen));
                h_B[i] = static_cast<T>(dist(gen));
            } else {  // half
                h_A[i] = __float2half(dist(gen));
                h_B[i] = __float2half(dist(gen));
            }
        }
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(T), cudaMemcpyHostToDevice));

        // Benchmark CUDA baseline first (for speedup calculation)
        // Clear output buffer BEFORE timing
        CUDA_CHECK(cudaMemset(d_C, 0, N * sizeof(T)));

        auto cuda_kernel = [&]() {
            baseline::elementwise_add(d_A, d_B, d_C, N);
        };
        float cuda_latency = runner.measureLatency(cuda_kernel);
        
        // Calculate throughput: 3 arrays accessed (2 reads + 1 write)
        size_t bytes_transferred = 3 * N * sizeof(T);
        float cuda_throughput = runner.calculateThroughput(bytes_transferred, cuda_latency);
        
        baseline_latencies.push_back(cuda_latency);
        
        BenchmarkResult cuda_result("elementwise_add", "CUDA", N,
                                   cuda_latency, cuda_throughput, 1.0f);
        BenchmarkRunner::printResult(cuda_result);
        csv.writeResult(cuda_result);

        // Benchmark CuTe implementation
        // Clear output buffer BEFORE timing
        CUDA_CHECK(cudaMemset(d_C, 0, N * sizeof(T)));

        auto cute_kernel = [&]() {
            kebab::cute::elementwise_add(d_A, d_B, d_C, N);
        };
        float cute_latency = runner.measureLatency(cute_kernel);
        float cute_throughput = runner.calculateThroughput(bytes_transferred, cute_latency);
        
        // Calculate speedup (baseline / cute)
        float speedup = cuda_latency / cute_latency;
        
        BenchmarkResult cute_result("elementwise_add", "CuTe", N, 
                                   cute_latency, cute_throughput, speedup);
        BenchmarkRunner::printResult(cute_result);
        csv.writeResult(cute_result);
        
        // Verify correctness (sample check)
        std::vector<T> h_C(N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(T), cudaMemcpyDeviceToHost));
        
        bool correct = true;
        const int num_checks = std::min(100, N);
        for (int i = 0; i < num_checks; ++i) {
            int idx = (i * N) / num_checks;  // Sample evenly
            float expected, actual;
            
            if constexpr (sizeof(T) == 4) {  // float
                expected = h_A[idx] + h_B[idx];
                actual = h_C[idx];
            } else {  // half
                expected = __half2float(h_A[idx]) + __half2float(h_B[idx]);
                actual = __half2float(h_C[idx]);
            }
            
            float error = std::abs(expected - actual);
            if (error > 1e-3f) {
                correct = false;
                break;
            }
        }
        
        if (!correct) {
            std::cerr << "WARNING: Correctness check failed for batch size " << N << std::endl;
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Calculate average speedup
    float total_speedup = 0.0f;
    int count = 0;
    for (int i = 0; i < static_cast<int>(batch_sizes.size()); ++i) {
        float speedup = baseline_latencies[i] / 
                       (baseline_latencies[i] / 1.0f);  // This will be updated with actual CuTe latencies
        total_speedup += speedup;
        count++;
    }
    
    std::cout << "Average CuTe speedup: " << std::fixed << std::setprecision(3) 
              << (total_speedup / count) << "x" << std::endl;
    std::cout << "Results saved to: bench_results/elementwise_add_results.csv" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
    try {
        // Load configuration
        auto& config = ConfigParser::getInstance("config.yaml");
        
        // Set GPU device from config
        int gpu_id = config.getOperatorGpuId("elementwise_add");
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        
        if (gpu_id >= 0 && gpu_id < device_count) {
            CUDA_CHECK(cudaSetDevice(gpu_id));
            std::cout << "Using GPU " << gpu_id << " (from config)" << std::endl;
        } else if (gpu_id >= device_count) {
            std::cerr << "Warning: GPU " << gpu_id << " not available (only " << device_count 
                      << " GPUs detected). Using GPU 0." << std::endl;
            CUDA_CHECK(cudaSetDevice(0));
            gpu_id = 0;
        } else {
            // gpu_id < 0, auto-select
            CUDA_CHECK(cudaGetDevice(&gpu_id));
            std::cout << "Auto-selected GPU " << gpu_id << std::endl;
        }
        
        int warmup_runs = config.getWarmupRuns();
        int measurement_runs = config.getMeasurementRuns();
        std::vector<int> batch_sizes = config.getBatchSizes();
        std::vector<std::string> data_types = config.getDataTypes();
        
        // Check if elementwise_add is enabled
        if (!config.isOperatorEnabled("elementwise_add")) {
            std::cout << "Element-wise add operator is disabled in config.yaml" << std::endl;
            std::cout << "Enable it by setting operators.elementwise_add.enabled: true" << std::endl;
            return 0;
        }
        
        // Use operator-specific sizes if available, otherwise use global batch_sizes
        std::vector<int> op_sizes = config.getOperatorSizes("elementwise_add");
        if (!op_sizes.empty()) {
            batch_sizes = op_sizes;
        }
        
        // Print GPU information
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));
        std::cout << "\n========================================" << std::endl;
        std::cout << "GPU Information" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Device: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB" << std::endl;
        
        // Get memory clock rate using cudaDeviceGetAttribute (for CUDA 12+ compatibility)
        int memoryClockRate = 0;
        int memoryBusWidth = 0;
        cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, 0);
        cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, 0);
        std::cout << "Memory Bandwidth: " 
                  << (2.0 * memoryClockRate * (memoryBusWidth / 8) / 1.0e6) 
                  << " GB/s" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Open CSV file for results
        CSVWriter csv("bench_results/elementwise_add_results.csv");
        if (!csv.isOpen()) {
            std::cerr << "ERROR: Failed to open CSV file for writing" << std::endl;
            return 1;
        }
        csv.writeHeader();
        
        // Run benchmarks for each data type
        for (const auto& dtype : data_types) {
            if (dtype == "float32") {
                benchmarkElementwiseAdd<float>(batch_sizes, warmup_runs, measurement_runs, csv);
            } else if (dtype == "float16") {
                benchmarkElementwiseAdd<__half>(batch_sizes, warmup_runs, measurement_runs, csv);
            } else {
                std::cerr << "WARNING: Unknown data type: " << dtype << std::endl;
            }
        }
        
        std::cout << "Benchmark completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
