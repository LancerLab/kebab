/**
 * @file kebab.h
 * @brief Main header file for the Kebab high-performance kernel library
 * 
 * This header provides access to all core functionality of the Kebab library,
 * including GEMM operations, element-wise operations, and utility functions.
 */

#pragma once

// CuTe operators
#include "kebab/cute/gemm.h"
#include "kebab/cute/elementwise_add.h"

// Utilities
#include "kebab/utils/matrix_init.h"
#include "kebab/utils/matrix_print.h"

// Configuration
#include "kebab/config/config_parser.h"

/**
 * @namespace kebab
 * @brief Main namespace for the Kebab library
 * 
 * The Kebab library provides high-performance GPU kernels using NVIDIA CuTe
 * and WGMMA instructions for Hopper architecture GPUs.
 */
namespace kebab {

/**
 * @brief Library version information
 */
struct Version {
    static constexpr int MAJOR = 1;
    static constexpr int MINOR = 0;
    static constexpr int PATCH = 0;
    
    static const char* getString() {
        return "1.0.0";
    }
};

} // namespace kebab