#include "cutekernellib/config/config_parser.h"
#include <iostream>
#include <iomanip>

using namespace cutekernellib::config;

void printSeparator() {
    std::cout << std::string(60, '=') << std::endl;
}

void printVector(const std::string& name, const std::vector<int>& vec) {
    std::cout << "  " << std::setw(25) << std::left << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void printVector(const std::string& name, const std::vector<std::string>& vec) {
    std::cout << "  " << std::setw(25) << std::left << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    std::string config_path = "config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    std::cout << "\n";
    printSeparator();
    std::cout << "CuTeKernelLib Configuration Parser Test" << std::endl;
    printSeparator();
    std::cout << "\n";
    
    try {
        // Get singleton instance and load config
        std::cout << "Loading configuration from: " << config_path << std::endl;
        ConfigParser& config = ConfigParser::getInstance(config_path);
        std::cout << "\n";
        
        // Test Build Configuration
        printSeparator();
        std::cout << "BUILD CONFIGURATION" << std::endl;
        printSeparator();
        std::cout << "  " << std::setw(25) << std::left << "Build Mode" << ": " << config.getBuildMode() << std::endl;
        std::cout << "  " << std::setw(25) << std::left << "Optimization Level" << ": " << config.getOptimizationLevel() << std::endl;
        std::cout << "  " << std::setw(25) << std::left << "CUDA Architecture" << ": " << config.getCudaArch() << std::endl;
        std::cout << "\n";
        
        // Test Benchmark Configuration
        printSeparator();
        std::cout << "BENCHMARK CONFIGURATION" << std::endl;
        printSeparator();
        std::cout << "  " << std::setw(25) << std::left << "Warmup Runs" << ": " << config.getWarmupRuns() << std::endl;
        std::cout << "  " << std::setw(25) << std::left << "Measurement Runs" << ": " << config.getMeasurementRuns() << std::endl;
        printVector("Batch Sizes", config.getBatchSizes());
        printVector("Data Types", config.getDataTypes());
        std::cout << "\n";
        
        // Test Profiling Configuration
        printSeparator();
        std::cout << "PROFILING CONFIGURATION" << std::endl;
        printSeparator();
        auto metrics = config.getProfilingMetrics();
        std::cout << "  NCU Metrics (" << metrics.size() << " total):" << std::endl;
        for (const auto& metric : metrics) {
            std::cout << "    - " << metric << std::endl;
        }
        printVector("Profiling Sections", config.getProfilingSections());
        std::cout << "\n";
        
        // Test Operator Configuration
        printSeparator();
        std::cout << "OPERATOR CONFIGURATION" << std::endl;
        printSeparator();
        
        std::vector<std::string> operators = {"elementwise_add", "gemm", "fft", "conv2d", "reduction"};
        for (const auto& op : operators) {
            std::cout << "  " << op << ":" << std::endl;
            std::cout << "    Enabled: " << (config.isOperatorEnabled(op) ? "true" : "false") << std::endl;
            
            auto sizes = config.getOperatorSizes(op);
            if (!sizes.empty()) {
                std::cout << "    Sizes: [";
                for (size_t i = 0; i < sizes.size(); ++i) {
                    std::cout << sizes[i];
                    if (i < sizes.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            auto tile_sizes = config.getOperatorTileSizes(op);
            if (!tile_sizes.empty()) {
                std::cout << "    Tile Sizes: [";
                for (size_t i = 0; i < tile_sizes.size(); ++i) {
                    std::cout << tile_sizes[i];
                    if (i < tile_sizes.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            auto matrix_sizes = config.getOperatorMatrixSizes(op);
            if (!matrix_sizes.empty()) {
                std::cout << "    Matrix Sizes: [";
                for (size_t i = 0; i < matrix_sizes.size(); ++i) {
                    std::cout << matrix_sizes[i];
                    if (i < matrix_sizes.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }
        
        // Success message
        printSeparator();
        std::cout << "✓ Configuration parser test PASSED" << std::endl;
        std::cout << "✓ All methods executed successfully" << std::endl;
        std::cout << "✓ Config file: " << config.getConfigPath() << std::endl;
        printSeparator();
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n";
        printSeparator();
        std::cerr << "✗ Configuration parser test FAILED" << std::endl;
        printSeparator();
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "\n";
        return 1;
    }
}
