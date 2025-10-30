#include "benchmark_runner.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

namespace cutekernellib {
namespace benchmark {

// BenchmarkRunner implementation

BenchmarkRunner::BenchmarkRunner(int warmup_runs, int measurement_runs)
    : warmup_runs_(warmup_runs), measurement_runs_(measurement_runs) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
}

BenchmarkRunner::~BenchmarkRunner() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

float BenchmarkRunner::calculateThroughput(size_t bytes_transferred, float latency_ms) const {
    // Convert to GB/s: (bytes / 1e9) / (latency_ms / 1e3)
    // Simplifies to: bytes / (latency_ms * 1e6)
    if (latency_ms <= 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(bytes_transferred) / (latency_ms * 1e6f);
}

float BenchmarkRunner::calculateGFLOPS(size_t flops, float latency_ms) const {
    // Convert to GFLOPS: (flops / 1e9) / (latency_ms / 1e3)
    // Simplifies to: flops / (latency_ms * 1e6)
    if (latency_ms <= 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(flops) / (latency_ms * 1e6f);
}

void BenchmarkRunner::printHeader() {
    std::cout << std::left
              << std::setw(20) << "Operator"
              << std::setw(12) << "Variant"
              << std::setw(12) << "Batch Size"
              << std::setw(15) << "Latency (ms)"
              << std::setw(18) << "Throughput (GFLOPS)"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(89, '-') << std::endl;
}

void BenchmarkRunner::printResult(const BenchmarkResult& result) {
    std::cout << std::left
              << std::setw(20) << result.operator_name
              << std::setw(12) << result.variant
              << std::setw(12) << result.batch_size
              << std::setw(15) << std::fixed << std::setprecision(6) << result.latency_ms
              << std::setw(18) << std::fixed << std::setprecision(2) << result.throughput_gbps
              << std::setw(10) << std::fixed << std::setprecision(3) << result.speedup_ratio << "x"
              << std::endl;
}

// CSVWriter implementation

CSVWriter::CSVWriter(const std::string& filename) : filename_(filename) {
    // Create bench_results directory if it doesn't exist
    struct stat st;
    if (stat("bench_results", &st) != 0) {
        #ifdef _WIN32
        _mkdir("bench_results");
        #else
        mkdir("bench_results", 0755);
        #endif
    }
    
    file_.open(filename, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        std::cerr << "ERROR: Failed to open CSV file: " << filename << std::endl;
    }
}

CSVWriter::~CSVWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

void CSVWriter::writeHeader() {
    if (!file_.is_open()) {
        return;
    }
    file_ << "Operator,Variant,BatchSize,Latency(ms),Throughput(GFLOPS),Speedup\n";
}

void CSVWriter::writeResult(const BenchmarkResult& result) {
    if (!file_.is_open()) {
        return;
    }
    file_ << result.operator_name << ","
          << result.variant << ","
          << result.batch_size << ","
          << std::fixed << std::setprecision(6) << result.latency_ms << ","
          << std::fixed << std::setprecision(2) << result.throughput_gbps << ","
          << std::fixed << std::setprecision(3) << result.speedup_ratio << "\n";
}

bool CSVWriter::isOpen() const {
    return file_.is_open();
}

} // namespace benchmark
} // namespace cutekernellib
