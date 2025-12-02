#include "kebab/utils/benchmark.h"
#include "kebab/cute/gemm.h"
#include "kebab/utils/matrix_init.h"
#include "kebab/utils/matrix_print.h"
#include "kebab/utils/cublas_reference.h"
#include "kebab/cuda/cuda_gemm.h"
#include "kebab/config/config_parser.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include <iomanip>
#include <sstream>
#include <algorithm>

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_RED     "\033[31m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_DIM     "\033[2m"

using namespace kebab::benchmark;
using namespace kebab::utils;
using namespace kebab::config;



/**
 * @brief Verify GEMM correctness against cuBLAS reference (wrapper for shared function)
 */
template<typename T>
bool verifyGEMM(const T* A, const T* B, const T* C_test, int M, int N, int K,
                const std::string& opmode, float tolerance = 1e-3f,
                bool /* transpose_compare */ = false) {
    char lhs_format = (opmode.length() >= 1) ? opmode[0] : 'R';
    char rhs_format = (opmode.length() >= 2) ? opmode[1] : 'R';
    // Use shared verification function (verbose=false for benchmark)
    return verifyCublasGemm(A, B, C_test, M, N, K, lhs_format, rhs_format, tolerance, false);
}

/**
 * @brief Helper to format latency with appropriate precision
 */
inline std::string formatLatency(float lat) {
    std::ostringstream ss;
    if (lat < 0.01f) ss << std::fixed << std::setprecision(4) << lat;
    else if (lat < 0.1f) ss << std::fixed << std::setprecision(3) << lat;
    else if (lat < 1.0f) ss << std::fixed << std::setprecision(2) << lat;
    else ss << std::fixed << std::setprecision(1) << lat;
    return ss.str();
}

/**
 * @brief Repeat a string n times (for UTF-8 box drawing characters)
 */
inline std::string repeatStr(const std::string& s, int n) {
    std::string result;
    for (int i = 0; i < n; ++i) result += s;
    return result;
}

/**
 * @brief Helper to format TFLOPS with color based on percentage of cuBLAS
 */
inline std::string formatTFLOPS(float tflops, float cublas_tflops) {
    std::ostringstream ss;
    float pct = (cublas_tflops > 0) ? (tflops / cublas_tflops * 100.0f) : 0.0f;
    const char* color = (pct >= 100.0f) ? COLOR_GREEN : (pct >= 70.0f) ? COLOR_YELLOW : COLOR_RED;
    ss << color << std::fixed << std::setprecision(1) << std::setw(6) << tflops << COLOR_RESET;
    return ss.str();
}

/**
 * @brief Benchmark GEMM with configuration from config.yaml
 *
 * Display hierarchy:
 *   Problem (precision) -> Size -> Mode -> Impl -> Versions (as table columns)
 */
template<typename T>
void benchmarkGEMM(const ConfigParser& config) {
    std::string type_name;
    if constexpr (std::is_same_v<T, float>) {
        type_name = "float32";
    } else if constexpr (std::is_same_v<T, __half>) {
        type_name = "float16";
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        type_name = "bfloat16";
    }

    // Get configuration
    auto impl_list = config.getOperatorImpls("gemm");
    auto version_list = config.getOperatorVersions("gemm");
    std::string init_method = config.getOperatorInitMethod("gemm");
    std::vector<std::string> opmode_list = config.getOperatorModes("gemm");
    auto matrix_sizes = config.getOperatorMatrixSizes("gemm");
    int warmup_runs = config.getWarmupRuns();
    int measurement_runs = config.getMeasurementRuns();
    bool verbose = config.getOperatorVerbose("gemm");

    // Parse init method
    auto [init_A, init_B] = parseBinaryInitMethod(init_method);

    // Print problem header
    std::cout << "\n" << COLOR_BOLD << COLOR_CYAN
              << "+" << repeatStr("=", 66) << "+" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << COLOR_CYAN << "|" << COLOR_RESET
              << "  GEMM Benchmark - " << COLOR_BOLD << type_name << COLOR_RESET
              << repeatStr(" ", 45 - static_cast<int>(type_name.length()))
              << COLOR_BOLD << COLOR_CYAN << "|" << COLOR_RESET << std::endl;
    std::cout << COLOR_BOLD << COLOR_CYAN
              << "+" << repeatStr("=", 66) << "+" << COLOR_RESET << std::endl;

    // Print config summary
    std::cout << COLOR_DIM << "  Impls: ";
    for (size_t i = 0; i < impl_list.size(); ++i) {
        std::cout << impl_list[i];
        if (i < impl_list.size() - 1) std::cout << ", ";
    }
    std::cout << " | Versions: ";
    for (size_t i = 0; i < version_list.size(); ++i) {
        std::cout << "v" << version_list[i];
        if (i < version_list.size() - 1) std::cout << ", ";
    }
    std::cout << " | Modes: ";
    for (size_t i = 0; i < opmode_list.size(); ++i) {
        std::cout << opmode_list[i];
        if (i < opmode_list.size() - 1) std::cout << ", ";
    }
    std::cout << COLOR_RESET << std::endl;
    
    BenchmarkRunner runner(warmup_runs, measurement_runs);

    // Create CSV writer with all impls in filename
    std::string impl_str;
    for (size_t i = 0; i < impl_list.size(); ++i) {
        impl_str += impl_list[i];
        if (i < impl_list.size() - 1) impl_str += "_";
    }
    std::string csv_filename = "bench_results/gemm_results_" + type_name + "_" + impl_str + ".csv";
    CSVWriter csv(csv_filename);
    if (!csv.isOpen()) {
        std::cerr << "ERROR: Failed to create CSV file: " << csv_filename << std::endl;
        return;
    }
    csv.writeHeader();

    // Random number generator for initialization
    std::mt19937 gen(42); // Fixed seed for reproducibility

    // Calculate column widths for table
    int col_width = 12;  // Width for each version column

    // Loop order: Size -> Mode -> Impl -> (Versions displayed as table)
    for (int size : matrix_sizes) {
        int M = size, N = size, K = size; // Square matrices

        // Size section header
        std::string size_str = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        int header_padding = 52 - static_cast<int>(size_str.length());
        std::cout << "\n" << COLOR_BOLD << "+-- Size: " << size_str << " "
                  << repeatStr("-", header_padding) << "+" << COLOR_RESET << std::endl;

        for (const auto& opmode : opmode_list) {
            // Parse opmode to determine storage formats
            char lhs_format = (opmode.length() >= 1) ? opmode[0] : 'R';
            char rhs_format = (opmode.length() >= 2) ? opmode[1] : 'R';

            // Storage dimensions
            int A_storage_rows = (lhs_format == 'R') ? M : K;
            int A_storage_cols = (lhs_format == 'R') ? K : M;
            int B_storage_rows = (rhs_format == 'R') ? K : N;
            int B_storage_cols = (rhs_format == 'R') ? N : K;

            // Allocate host memory
            std::vector<T> h_A(A_storage_rows * A_storage_cols);
            std::vector<T> h_B(B_storage_rows * B_storage_cols);

            // Initialize matrices
            initializeMatrix(h_A.data(), M, K, init_A, gen, lhs_format);
            initializeMatrix(h_B.data(), K, N, init_B, gen, rhs_format);

            // Allocate device memory
            T *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, A_storage_rows * A_storage_cols * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_B, B_storage_rows * B_storage_cols * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(T)));

            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_storage_rows * A_storage_cols * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_storage_rows * B_storage_cols * sizeof(T), cudaMemcpyHostToDevice));

            // First: benchmark cuBLAS (reference for all)
            size_t flops = 2ULL * M * N * K;
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));

            // Get cuBLAS config using already-parsed lhs_format and rhs_format
            auto cublas_config = getCublasGemmConfig(lhs_format, rhs_format, M, N, K);

            cublasHandle_t handle;
            cublasCreate(&handle);

            // Use shared cuBLAS function for benchmarking
            auto cublas_kernel = [&]() {
                runCublasGemm(handle, d_A, d_B, d_C, M, N, K, cublas_config);
            };

            float cublas_latency = runner.measureLatency(cublas_kernel);
            cublasDestroy(handle);
            float cublas_tflops = runner.calculateGFLOPS(flops, cublas_latency) / 1000.0f;

            // Mode header with table structure
            std::cout << COLOR_DIM << "|" << COLOR_RESET << " Mode: " << COLOR_BOLD << opmode << COLOR_RESET << std::endl;

            // Print table header: | Impl | cuBLAS | v1 | v2 | v3 | ... |
            std::cout << COLOR_DIM << "|  +";
            std::cout << repeatStr("-", 10) << "+" << repeatStr("-", col_width);  // Impl + cuBLAS
            for (size_t i = 0; i < version_list.size(); ++i) {
                std::cout << "+" << repeatStr("-", col_width);
            }
            std::cout << "+" << COLOR_RESET << std::endl;

            // Header row
            std::cout << COLOR_DIM << "|  |" << COLOR_RESET << std::setw(10) << "Impl"
                      << COLOR_DIM << "|" << COLOR_RESET << std::setw(col_width) << "cuBLAS";
            for (int ver : version_list) {
                std::cout << COLOR_DIM << "|" << COLOR_RESET << std::setw(col_width) << ("v" + std::to_string(ver));
            }
            std::cout << COLOR_DIM << "|" << COLOR_RESET << std::endl;

            // Separator
            std::cout << COLOR_DIM << "|  +";
            std::cout << repeatStr("-", 10) << "+" << repeatStr("-", col_width);
            for (size_t i = 0; i < version_list.size(); ++i) {
                std::cout << "+" << repeatStr("-", col_width);
            }
            std::cout << "+" << COLOR_RESET << std::endl;

            // For each impl, run all versions and display as row
            for (const auto& impl : impl_list) {
                // Skip CuTe for bfloat16
                if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    if (impl == "cute") continue;
                }

                // Collect TFLOPS for each version
                std::vector<float> version_tflops;
                std::vector<bool> version_valid;

                // Use shared tolerance calculation for consistency with runonce
                float tolerance = getVerificationTolerance<T>(size);

                for (int version : version_list) {
                    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));

                    auto kernel = [&]() {
                        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                            baseline::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
                        } else {
                            if (impl == "cute") {
                                kebab::cute::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
                            } else {
                                baseline::gemm(d_A, d_B, d_C, M, N, K, opmode.c_str(), version);
                            }
                        }
                    };

                    // Verify first
                    kernel();
                    bool correct = verifyGEMM(d_A, d_B, d_C, M, N, K, opmode, tolerance, false);

                    if (correct) {
                        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(T)));
                        float latency = runner.measureLatency(kernel);
                        float tflops = runner.calculateGFLOPS(flops, latency) / 1000.0f;
                        version_tflops.push_back(tflops);
                        version_valid.push_back(true);

                        // Save to CSV
                        BenchmarkResult result;
                        result.operator_name = "GEMM";
                        result.variant = impl + "_v" + std::to_string(version);
                        result.batch_size = size;
                        result.latency_ms = latency;
                        result.throughput_gbps = tflops * 1000.0f;
                        result.speedup_ratio = tflops / cublas_tflops;
                        csv.writeResult(result);
                    } else {
                        version_tflops.push_back(0.0f);
                        version_valid.push_back(false);
                    }
                }

                // Print row: | impl | cuBLAS_TFLOPS | v1_TFLOPS | v2_TFLOPS | ... |
                std::cout << COLOR_DIM << "|  |" << COLOR_RESET << std::setw(10) << impl
                          << COLOR_DIM << "|" << COLOR_RESET;

                // cuBLAS column (always reference)
                std::cout << std::fixed << std::setprecision(1) << std::setw(col_width) << cublas_tflops;

                // Version columns with color coding and percentage
                for (size_t i = 0; i < version_list.size(); ++i) {
                    std::cout << COLOR_DIM << "|" << COLOR_RESET;
                    if (!version_valid[i]) {
                        std::cout << COLOR_RED << std::setw(col_width) << "FAIL" << COLOR_RESET;
                    } else {
                        float pct = (cublas_tflops > 0) ? (version_tflops[i] / cublas_tflops * 100.0f) : 0.0f;
                        const char* color = (pct >= 100.0f) ? COLOR_GREEN : (pct >= 70.0f) ? COLOR_YELLOW : COLOR_RED;
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(1) << version_tflops[i]
                            << "(" << static_cast<int>(pct) << "%)";
                        std::cout << color << std::setw(col_width) << oss.str() << COLOR_RESET;
                    }
                }
                std::cout << COLOR_DIM << "|" << COLOR_RESET << std::endl;
            }

            // Table footer
            std::cout << COLOR_DIM << "|  +";
            std::cout << repeatStr("-", 10) << "+" << repeatStr("-", col_width);
            for (size_t i = 0; i < version_list.size(); ++i) {
                std::cout << "+" << repeatStr("-", col_width);
            }
            std::cout << "+" << COLOR_RESET << std::endl;

            // Cleanup device memory for this mode
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }

        // Size section footer
        std::cout << COLOR_BOLD << "+" << repeatStr("-", 65) << "+" << COLOR_RESET << std::endl;
    }

    std::cout << "\n" << COLOR_GREEN << "✓ Benchmark complete!" << COLOR_RESET
              << " Results saved to: " << csv_filename << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "CuTeKernelLib GEMM Benchmark" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Load configuration
    auto& config = ConfigParser::getInstance("config.yaml");
    std::cout << "Configuration loaded successfully from: config.yaml" << std::endl;
    
    // Set GPU device from config
    int gpu_id = config.getOperatorGpuId("gemm");
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
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  GPU ID:           " << gpu_id << std::endl;
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

    // Get implementations from config
    auto impl_list = config.getOperatorImpls("gemm");
    bool has_cuda_impl = std::find(impl_list.begin(), impl_list.end(), "cuda") != impl_list.end();

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
        } else if (precision == "bfloat16" || precision == "bf16") {
            if (!has_tensor_cores) {
                std::cout << "\nSkipping bfloat16 precision benchmark (requires Tensor Core support)" << std::endl;
            } else if (!has_cuda_impl) {
                std::cout << "\nSkipping bfloat16 precision benchmark (CuTe does not support bfloat16, add impl: cuda)" << std::endl;
            } else {
                benchmarkGEMM<__nv_bfloat16>(config);
            }
        }
    }

    return 0;
}
