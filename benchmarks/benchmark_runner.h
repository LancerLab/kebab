#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace cutekernellib {
namespace benchmark {

/**
 * @brief Structure to store benchmark results for a single test
 */
struct BenchmarkResult {
    std::string operator_name;      // Name of the operator (e.g., "elementwise_add")
    std::string variant;            // Implementation variant ("CuTe" or "CUDA")
    int batch_size;                 // Size of the input data
    float latency_ms;               // Average latency in milliseconds
    float throughput_gbps;          // Throughput in GB/s
    float speedup_ratio;            // Speedup compared to baseline (1.0 for baseline)
    
    BenchmarkResult() 
        : batch_size(0), latency_ms(0.0f), throughput_gbps(0.0f), speedup_ratio(1.0f) {}
    
    BenchmarkResult(const std::string& op, const std::string& var, int size, 
                   float latency, float throughput, float speedup = 1.0f)
        : operator_name(op), variant(var), batch_size(size), 
          latency_ms(latency), throughput_gbps(throughput), speedup_ratio(speedup) {}
};

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            fprintf(stderr, "Error code: %d\n", err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Benchmark runner class for timing GPU kernels with CUDA events
 * 
 * This class provides infrastructure for benchmarking GPU kernels with:
 * - Warmup phase to stabilize GPU clocks
 * - Multiple measurement runs for statistical reliability
 * - CUDA event-based timing for microsecond precision
 * - Throughput calculation for bandwidth-bound operations
 */
class BenchmarkRunner {
public:
    /**
     * @brief Constructor
     * @param warmup_runs Number of warmup iterations before measurement
     * @param measurement_runs Number of measurement iterations for averaging
     */
    BenchmarkRunner(int warmup_runs, int measurement_runs);
    
    /**
     * @brief Destructor - cleans up CUDA events
     */
    ~BenchmarkRunner();
    
    /**
     * @brief Measure latency of a kernel function
     * @param kernel_func Function object that launches the kernel
     * @return Average latency in milliseconds
     * 
     * The function performs warmup runs followed by measurement runs,
     * using CUDA events for precise timing.
     */
    template<typename Func>
    float measureLatency(Func kernel_func) {
        // Warmup phase - stabilize GPU clocks and caches
        for (int i = 0; i < warmup_runs_; ++i) {
            kernel_func();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Measurement phase
        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < measurement_runs_; ++i) {
            kernel_func();
        }
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        
        // Calculate average latency
        float total_ms;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_, stop_));
        return total_ms / measurement_runs_;
    }
    
    /**
     * @brief Calculate throughput in GB/s for memory-bound operations
     * @param bytes_transferred Total bytes read + written by the kernel
     * @param latency_ms Kernel latency in milliseconds
     * @return Throughput in GB/s
     */
    float calculateThroughput(size_t bytes_transferred, float latency_ms) const;
    
    /**
     * @brief Calculate GFLOPS for compute-bound operations
     * @param flops Number of floating-point operations
     * @param latency_ms Kernel latency in milliseconds
     * @return Performance in GFLOPS
     */
    float calculateGFLOPS(size_t flops, float latency_ms) const;
    
    /**
     * @brief Get number of warmup runs
     */
    int getWarmupRuns() const { return warmup_runs_; }
    
    /**
     * @brief Get number of measurement runs
     */
    int getMeasurementRuns() const { return measurement_runs_; }
    
    /**
     * @brief Print benchmark result to console
     */
    static void printResult(const BenchmarkResult& result);
    
    /**
     * @brief Print benchmark header to console
     */
    static void printHeader();

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    int warmup_runs_;
    int measurement_runs_;
};

/**
 * @brief Helper class for CSV output
 */
class CSVWriter {
public:
    /**
     * @brief Constructor
     * @param filename Output CSV file path
     */
    explicit CSVWriter(const std::string& filename);
    
    /**
     * @brief Destructor - closes file
     */
    ~CSVWriter();
    
    /**
     * @brief Write CSV header
     */
    void writeHeader();
    
    /**
     * @brief Write a benchmark result row
     */
    void writeResult(const BenchmarkResult& result);
    
    /**
     * @brief Check if file is open
     */
    bool isOpen() const;

private:
    std::ofstream file_;
    std::string filename_;
};

} // namespace benchmark
} // namespace cutekernellib
