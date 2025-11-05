#include "benchmark_runner.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace kebab::benchmark;

// Simple dummy kernel for testing
__global__ void dummyKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Simple computation to ensure kernel does some work
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    std::cout << "Testing BenchmarkRunner with dummy kernel..." << std::endl;
    std::cout << std::endl;
    
    // Test parameters
    const int warmup = 5;
    const int measurement = 20;
    const int size = 1024 * 1024;  // 1M elements
    
    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    // Initialize data
    std::vector<float> h_data(size, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create benchmark runner
    BenchmarkRunner runner(warmup, measurement);
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Warmup runs: " << runner.getWarmupRuns() << std::endl;
    std::cout << "  Measurement runs: " << runner.getMeasurementRuns() << std::endl;
    std::cout << "  Data size: " << size << " elements (" 
              << (size * sizeof(float)) / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << std::endl;
    
    // Define kernel launch parameters
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    // Measure latency
    std::cout << "Running benchmark..." << std::endl;
    auto kernel_func = [&]() {
        dummyKernel<<<grid, block>>>(d_data, size);
    };
    
    float latency_ms = runner.measureLatency(kernel_func);
    
    // Calculate throughput (read + write = 2 * size * sizeof(float))
    size_t bytes_transferred = 2 * size * sizeof(float);
    float throughput_gbps = runner.calculateThroughput(bytes_transferred, latency_ms);
    
    // Create and print result
    BenchmarkResult result("dummy_kernel", "test", size, latency_ms, throughput_gbps, 1.0f);
    
    std::cout << std::endl;
    BenchmarkRunner::printHeader();
    BenchmarkRunner::printResult(result);
    
    std::cout << std::endl;
    std::cout << "Detailed metrics:" << std::endl;
    std::cout << "  Average latency: " << latency_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput_gbps << " GB/s" << std::endl;
    std::cout << "  Bytes transferred: " << bytes_transferred << " bytes" << std::endl;
    
    // Verify timing is reasonable (should be > 0 and < 1 second for this simple kernel)
    if (latency_ms > 0.0f && latency_ms < 1000.0f) {
        std::cout << std::endl;
        std::cout << "✓ Timing appears reasonable!" << std::endl;
    } else {
        std::cout << std::endl;
        std::cout << "✗ WARNING: Timing seems unusual!" << std::endl;
    }
    
    // Test CSV writer
    std::cout << std::endl;
    std::cout << "Testing CSV writer..." << std::endl;
    CSVWriter csv("bench_results/test_benchmark_runner.csv");
    if (csv.isOpen()) {
        csv.writeHeader();
        csv.writeResult(result);
        std::cout << "✓ CSV file written successfully to bench_results/test_benchmark_runner.csv" << std::endl;
    } else {
        std::cout << "✗ Failed to open CSV file" << std::endl;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    
    std::cout << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}
