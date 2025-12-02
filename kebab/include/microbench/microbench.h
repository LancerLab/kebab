#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace kebab {
namespace microbench {

/**
 * @brief CUDA error checking macro for microbenchmarks
 */
#define MBENCH_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Result structure for a single microbenchmark variant
 */
struct MicrobenchResult {
    std::string variant_name;       // Name of the implementation variant
    std::string description;        // Brief description of the implementation
    size_t data_size_bytes;         // Size of data transferred
    float latency_us;               // Average latency in microseconds
    float bandwidth_gbps;           // Achieved bandwidth in GB/s
    float efficiency_pct;           // Efficiency as percentage of peak bandwidth
    bool is_baseline;               // Whether this is the baseline variant
    
    MicrobenchResult() 
        : data_size_bytes(0), latency_us(0.0f), bandwidth_gbps(0.0f), 
          efficiency_pct(0.0f), is_baseline(false) {}
};

/**
 * @brief Microbenchmark runner with precise CUDA event timing
 */
class MicrobenchRunner {
public:
    MicrobenchRunner(int warmup_iters = 10, int measure_iters = 100)
        : warmup_iters_(warmup_iters), measure_iters_(measure_iters) {
        MBENCH_CUDA_CHECK(cudaEventCreate(&start_));
        MBENCH_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~MicrobenchRunner() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    /**
     * @brief Measure kernel latency in microseconds
     */
    template<typename Func>
    float measureLatencyUs(Func kernel_func) {
        // Warmup
        for (int i = 0; i < warmup_iters_; ++i) {
            kernel_func();
        }
        MBENCH_CUDA_CHECK(cudaDeviceSynchronize());
        
        // Measurement
        MBENCH_CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < measure_iters_; ++i) {
            kernel_func();
        }
        MBENCH_CUDA_CHECK(cudaEventRecord(stop_));
        MBENCH_CUDA_CHECK(cudaEventSynchronize(stop_));
        
        float total_ms;
        MBENCH_CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_, stop_));
        return (total_ms * 1000.0f) / measure_iters_;  // Convert to microseconds
    }
    
    /**
     * @brief Calculate bandwidth in GB/s
     */
    static float calculateBandwidthGBps(size_t bytes, float latency_us) {
        if (latency_us <= 0.0f) return 0.0f;
        // bytes / latency_us = bytes/us = MB/s, divide by 1000 to get GB/s
        return static_cast<float>(bytes) / (latency_us * 1000.0f);
    }
    
    int getWarmupIters() const { return warmup_iters_; }
    int getMeasureIters() const { return measure_iters_; }

private:
    cudaEvent_t start_, stop_;
    int warmup_iters_;
    int measure_iters_;
};

/**
 * @brief Report generator for structured microbenchmark output
 */
class MicrobenchReport {
public:
    MicrobenchReport(const std::string& bench_name, float peak_bandwidth_gbps)
        : bench_name_(bench_name), peak_bandwidth_gbps_(peak_bandwidth_gbps) {}
    
    void addResult(const MicrobenchResult& result) {
        results_.push_back(result);
    }
    
    void printHeader() const {
        std::cout << "\n";
        std::cout << "==========================================\n";
        std::cout << "Microbenchmark: " << bench_name_ << "\n";
        std::cout << "==========================================\n";
        printDeviceInfo();
        std::cout << "Peak Memory Bandwidth: " << std::fixed << std::setprecision(1) 
                  << peak_bandwidth_gbps_ << " GB/s\n";
        std::cout << "------------------------------------------\n";
    }
    
    void printTable() const;
    void printSummary() const;

private:
    void printDeviceInfo() const {
        cudaDeviceProp prop;
        MBENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Device: " << prop.name << "\n";
    }

    std::string formatBytes(size_t bytes) const;

    std::string bench_name_;
    float peak_bandwidth_gbps_;
    std::vector<MicrobenchResult> results_;
};

} // namespace microbench
} // namespace kebab

