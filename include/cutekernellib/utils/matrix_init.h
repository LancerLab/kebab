/**
 * @file matrix_init.h
 * @brief Matrix initialization utilities for testing and benchmarking
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <string>
#include <vector>

namespace cutekernellib {
namespace utils {

/**
 * @brief Initialization modes for matrices
 */
enum class InitMode {
    ONE,    // All elements are 1.0
    RAND,   // Random values in [-1, 1]
    RANGE,  // Sequential values [0, 1, 2, ...]
    ROW,    // Value equals row index
    COL     // Value equals column index
};

/**
 * @brief Parse init mode from string
 */
inline InitMode parseInitMode(const std::string& mode_str) {
    if (mode_str == "one") return InitMode::ONE;
    if (mode_str == "rand") return InitMode::RAND;
    if (mode_str == "range") return InitMode::RANGE;
    if (mode_str == "row") return InitMode::ROW;
    if (mode_str == "col") return InitMode::COL;
    return InitMode::RAND; // default
}

/**
 * @brief Initialize matrix on host with specified mode
 * 
 * @tparam T Data type (float or __half)
 * @param data Pointer to host memory
 * @param rows Number of logical rows
 * @param cols Number of logical columns
 * @param mode Initialization mode
 * @param gen Random number generator (for RAND mode)
 * @param storage_format 'R' for row-major (default), 'C' for column-major
 */
template<typename T>
void initializeMatrix(T* data, int rows, int cols, InitMode mode, std::mt19937& gen, char storage_format = 'R') {
    switch (mode) {
        case InitMode::ONE:
            for (int i = 0; i < rows * cols; ++i) {
                if constexpr (std::is_same_v<T, __half>) {
                    data[i] = __float2half(1.0f);
                } else {
                    data[i] = T(1.0);
                }
            }
            break;
            
        case InitMode::RAND: {
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int i = 0; i < rows * cols; ++i) {
                float val = dist(gen);
                if constexpr (std::is_same_v<T, __half>) {
                    data[i] = __float2half(val);
                } else {
                    data[i] = T(val);
                }
            }
            break;
        }
        
        case InitMode::RANGE:
            for (int i = 0; i < rows * cols; ++i) {
                if constexpr (std::is_same_v<T, __half>) {
                    data[i] = __float2half(static_cast<float>(i));
                } else {
                    data[i] = T(i);
                }
            }
            break;
            
        case InitMode::ROW:
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int idx = (storage_format == 'R') ? (i * cols + j) : (j * rows + i);
                    if constexpr (std::is_same_v<T, __half>) {
                        data[idx] = __float2half(static_cast<float>(i));
                    } else {
                        data[idx] = T(i);
                    }
                }
            }
            break;
            
        case InitMode::COL:
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int idx = (storage_format == 'R') ? (i * cols + j) : (j * rows + i);
                    if constexpr (std::is_same_v<T, __half>) {
                        data[idx] = __float2half(static_cast<float>(j));
                    } else {
                        data[idx] = T(j);
                    }
                }
            }
            break;
    }
}

/**
 * @brief Initialize vector (1D array) with specified mode
 */
template<typename T>
void initializeVector(T* data, int size, InitMode mode, std::mt19937& gen, char storage_format = 'R') {
    initializeMatrix(data, 1, size, mode, gen, storage_format);
}

/**
 * @brief Parse binary init method string (e.g., "rand-one")
 * 
 * @param init_method String in format "<lhs>-<rhs>"
 * @return Pair of (lhs_mode, rhs_mode)
 */
inline std::pair<InitMode, InitMode> parseBinaryInitMethod(const std::string& init_method) {
    size_t dash_pos = init_method.find('-');
    if (dash_pos == std::string::npos) {
        // No dash found, use same mode for both
        InitMode mode = parseInitMode(init_method);
        return {mode, mode};
    }
    
    std::string lhs_str = init_method.substr(0, dash_pos);
    std::string rhs_str = init_method.substr(dash_pos + 1);
    
    return {parseInitMode(lhs_str), parseInitMode(rhs_str)};
}

} // namespace utils
} // namespace cutekernellib
