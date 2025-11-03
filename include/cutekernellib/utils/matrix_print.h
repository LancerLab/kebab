/**
 * @file matrix_print.h
 * @brief Beautiful matrix printing utilities for debugging
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cmath>

namespace cutekernellib {
namespace utils {

/**
 * @brief Color codes for terminal output
 */
namespace Colors {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BLACK = "\033[30m";
    const std::string BOLD = "\033[1m";
    const std::string DIM = "\033[2m";
    
    // Background colors for highlighting errors
    const std::string BG_RED = "\033[41m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_GREEN = "\033[42m";
}

/**
 * @brief Configuration for matrix printing
 */
struct MatrixPrintConfig {
    int max_rows = 16;          // Maximum rows to print
    int max_cols = 16;          // Maximum columns to print
    int precision = 3;          // Decimal precision
    int width = 8;              // Field width
    bool show_indices = true;   // Show row/column indices
    bool use_colors = true;     // Use terminal colors
    float error_threshold = 1e-3f; // Threshold for highlighting errors
    bool compact_mode = false;  // Compact display for large matrices
    bool show_comparison = true; // Show side-by-side comparison
    int comparison_cols = 8;    // Max columns for side-by-side comparison
};

/**
 * @brief Convert value to float for printing
 */
template<typename T>
float toFloat(const T& val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else {
        return static_cast<float>(val);
    }
}

/**
 * @brief Format a single value with color coding
 */
template<typename T>
std::string formatValue(const T& val, const T& ref_val, const MatrixPrintConfig& config) {
    float v = toFloat(val);
    float r = toFloat(ref_val);
    float error = std::abs(v - r);
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(config.precision) << std::setw(config.width);
    
    if (config.use_colors && error > config.error_threshold) {
        // Highlight errors with background color
        if (error > config.error_threshold * 10) {
            oss << Colors::BG_RED << Colors::WHITE;  // Severe error
        } else {
            oss << Colors::BG_YELLOW << Colors::BLACK;  // Moderate error
        }
        oss << v << Colors::RESET;
    } else if (config.use_colors && error < config.error_threshold * 0.1f) {
        // Highlight correct values in green
        oss << Colors::GREEN << v << Colors::RESET;
    } else {
        oss << v;
    }
    
    return oss.str();
}

/**
 * @brief Print matrix header with column indices
 */
void printMatrixHeader(int cols, int start_col, const MatrixPrintConfig& config) {
    if (!config.show_indices) return;
    
    std::cout << "    ";  // Space for row indices
    for (int j = start_col; j < std::min(start_col + config.max_cols, cols); ++j) {
        std::cout << std::setw(config.width) << j;
    }
    std::cout << std::endl;
    
    // Print separator line
    std::cout << "    ";
    for (int j = start_col; j < std::min(start_col + config.max_cols, cols); ++j) {
        std::cout << std::string(config.width, '-');
    }
    std::cout << std::endl;
}

/**
 * @brief Print a section of the matrix with side-by-side comparison
 */
template<typename T>
void printMatrixSectionComparison(const std::vector<T>& result, const std::vector<T>& reference,
                                 int rows, int cols, int start_row, int start_col,
                                 const std::string& title, const MatrixPrintConfig& config) {
    
    std::cout << "\n" << Colors::BOLD << title << Colors::RESET << std::endl;
    std::cout << "Showing region [" << start_row << ":" << std::min(start_row + config.max_rows, rows) 
              << ", " << start_col << ":" << std::min(start_col + config.max_cols, cols) << "]" << std::endl;
    
    int display_cols = std::min(config.comparison_cols, std::min(config.max_cols, cols - start_col));
    
    // Print "Result" header
    std::cout << "\n" << Colors::CYAN << "Result:" << Colors::RESET << std::endl;
    if (config.show_indices) {
        std::cout << "    ";
        for (int j = start_col; j < start_col + display_cols; ++j) {
            std::cout << std::setw(config.width) << j;
        }
        std::cout << std::endl;
        std::cout << "    ";
        for (int j = 0; j < display_cols; ++j) {
            std::cout << std::string(config.width, '-');
        }
        std::cout << std::endl;
    }
    
    for (int i = start_row; i < std::min(start_row + config.max_rows, rows); ++i) {
        if (config.show_indices) {
            std::cout << std::setw(3) << i << ":";
        }
        
        for (int j = start_col; j < start_col + display_cols; ++j) {
            int idx = i * cols + j;
            float val = toFloat(result[idx]);
            float ref = toFloat(reference[idx]);
            float error = std::abs(val - ref);
            
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(config.precision) << std::setw(config.width);
            
            if (config.use_colors && error > config.error_threshold) {
                if (error > config.error_threshold * 10) {
                    oss << Colors::BG_RED << Colors::WHITE;
                } else {
                    oss << Colors::BG_YELLOW << Colors::BLACK;
                }
                oss << val << Colors::RESET;
            } else if (config.use_colors && error < config.error_threshold * 0.1f) {
                oss << Colors::GREEN << val << Colors::RESET;
            } else {
                oss << val;
            }
            
            std::cout << oss.str();
        }
        std::cout << std::endl;
    }
    
    // Print "Expected" header
    std::cout << "\n" << Colors::MAGENTA << "Expected:" << Colors::RESET << std::endl;
    if (config.show_indices) {
        std::cout << "    ";
        for (int j = start_col; j < start_col + display_cols; ++j) {
            std::cout << std::setw(config.width) << j;
        }
        std::cout << std::endl;
        std::cout << "    ";
        for (int j = 0; j < display_cols; ++j) {
            std::cout << std::string(config.width, '-');
        }
        std::cout << std::endl;
    }
    
    for (int i = start_row; i < std::min(start_row + config.max_rows, rows); ++i) {
        if (config.show_indices) {
            std::cout << std::setw(3) << i << ":";
        }
        
        for (int j = start_col; j < start_col + display_cols; ++j) {
            int idx = i * cols + j;
            float ref = toFloat(reference[idx]);
            std::cout << std::fixed << std::setprecision(config.precision) 
                     << std::setw(config.width) << ref;
        }
        std::cout << std::endl;
    }
    
    // Print "Difference" header
    std::cout << "\n" << Colors::YELLOW << "Difference (Result - Expected):" << Colors::RESET << std::endl;
    if (config.show_indices) {
        std::cout << "    ";
        for (int j = start_col; j < start_col + display_cols; ++j) {
            std::cout << std::setw(config.width) << j;
        }
        std::cout << std::endl;
        std::cout << "    ";
        for (int j = 0; j < display_cols; ++j) {
            std::cout << std::string(config.width, '-');
        }
        std::cout << std::endl;
    }
    
    for (int i = start_row; i < std::min(start_row + config.max_rows, rows); ++i) {
        if (config.show_indices) {
            std::cout << std::setw(3) << i << ":";
        }
        
        for (int j = start_col; j < start_col + display_cols; ++j) {
            int idx = i * cols + j;
            float val = toFloat(result[idx]);
            float ref = toFloat(reference[idx]);
            float diff = val - ref;
            
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(config.precision) << std::setw(config.width);
            
            if (config.use_colors) {
                if (std::abs(diff) > config.error_threshold * 10) {
                    oss << Colors::RED << diff << Colors::RESET;
                } else if (std::abs(diff) > config.error_threshold) {
                    oss << Colors::YELLOW << diff << Colors::RESET;
                } else {
                    oss << Colors::GREEN << diff << Colors::RESET;
                }
            } else {
                oss << diff;
            }
            
            std::cout << oss.str();
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Print a section of the matrix (original single matrix display)
 */
template<typename T>
void printMatrixSection(const std::vector<T>& matrix, const std::vector<T>& reference,
                       int rows, int cols, int start_row, int start_col,
                       const std::string& title, const MatrixPrintConfig& config) {
    
    if (config.show_comparison) {
        printMatrixSectionComparison(matrix, reference, rows, cols, start_row, start_col, title, config);
        return;
    }
    
    std::cout << "\n" << Colors::BOLD << title << Colors::RESET << std::endl;
    std::cout << "Showing region [" << start_row << ":" << std::min(start_row + config.max_rows, rows) 
              << ", " << start_col << ":" << std::min(start_col + config.max_cols, cols) << "]" << std::endl;
    
    printMatrixHeader(cols, start_col, config);
    
    for (int i = start_row; i < std::min(start_row + config.max_rows, rows); ++i) {
        if (config.show_indices) {
            std::cout << std::setw(3) << i << ":";
        }
        
        for (int j = start_col; j < std::min(start_col + config.max_cols, cols); ++j) {
            int idx = i * cols + j;
            std::cout << formatValue(matrix[idx], reference[idx], config);
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Find regions with the most errors for focused printing
 */
template<typename T>
std::vector<std::pair<int, int>> findErrorRegions(const std::vector<T>& matrix, 
                                                  const std::vector<T>& reference,
                                                  int rows, int cols, 
                                                  const MatrixPrintConfig& config) {
    std::vector<std::pair<int, int>> error_regions;
    
    // Divide matrix into blocks and count errors in each
    int block_size = 8;
    for (int bi = 0; bi < rows; bi += block_size) {
        for (int bj = 0; bj < cols; bj += block_size) {
            int error_count = 0;
            
            for (int i = bi; i < std::min(bi + block_size, rows); ++i) {
                for (int j = bj; j < std::min(bj + block_size, cols); ++j) {
                    int idx = i * cols + j;
                    float error = std::abs(toFloat(matrix[idx]) - toFloat(reference[idx]));
                    if (error > config.error_threshold) {
                        error_count++;
                    }
                }
            }
            
            if (error_count > 0) {
                error_regions.push_back({bi, bj});
            }
        }
    }
    
    return error_regions;
}

/**
 * @brief Print matrix comparison with beautiful formatting
 */
template<typename T>
void printMatrixComparison(const std::vector<T>& result, const std::vector<T>& reference,
                          int rows, int cols, const std::string& name,
                          const MatrixPrintConfig& config = MatrixPrintConfig{}) {
    
    std::cout << "\n" << Colors::BOLD << Colors::CYAN 
              << "═══════════════════════════════════════════════════════════════" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << Colors::CYAN << "  Matrix Comparison: " << name << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << Colors::CYAN 
              << "═══════════════════════════════════════════════════════════════" << Colors::RESET << std::endl;
    
    std::cout << "Matrix size: " << rows << " × " << cols << std::endl;
    
    // Count total errors
    int total_errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float error = std::abs(toFloat(result[i]) - toFloat(reference[i]));
        if (error > config.error_threshold) {
            total_errors++;
            max_error = std::max(max_error, error);
        }
    }
    
    std::cout << "Total errors: " << total_errors << " / " << (rows * cols) 
              << " (" << std::fixed << std::setprecision(2) 
              << (100.0f * total_errors / (rows * cols)) << "%)" << std::endl;
    std::cout << "Max error: " << std::fixed << std::setprecision(6) << max_error << std::endl;
    std::cout << "Error threshold: " << config.error_threshold << std::endl;
    
    if (config.use_colors) {
        if (config.show_comparison) {
            std::cout << "\nColor legend: " 
                      << Colors::GREEN << "Correct" << Colors::RESET << " | "
                      << Colors::YELLOW << "Small Error" << Colors::RESET << " | "
                      << Colors::RED << "Large Error" << Colors::RESET << std::endl;
        } else {
            std::cout << "\nColor legend: " 
                      << Colors::GREEN << "Correct" << Colors::RESET << " | "
                      << Colors::BG_YELLOW << Colors::BLACK << "Error" << Colors::RESET << " | "
                      << Colors::BG_RED << Colors::WHITE << "Severe" << Colors::RESET << std::endl;
        }
    }
    
    // For small matrices, print everything
    if (rows <= config.max_rows && cols <= config.max_cols) {
        printMatrixSection(result, reference, rows, cols, 0, 0, 
                          "Complete Matrix", config);
        return;
    }
    
    // For large matrices, print strategic sections
    std::cout << "\n" << Colors::YELLOW << "Matrix too large, showing key regions..." << Colors::RESET << std::endl;
    
    // Always show top-left corner
    printMatrixSection(result, reference, rows, cols, 0, 0, 
                      "Top-Left Corner", config);
    
    // Show error regions
    auto error_regions = findErrorRegions(result, reference, rows, cols, config);
    int regions_shown = 0;
    for (const auto& region : error_regions) {
        if (regions_shown >= 3) break;  // Limit to 3 error regions
        
        std::string title = "Error Region " + std::to_string(regions_shown + 1) + 
                           " (starting at [" + std::to_string(region.first) + 
                           "," + std::to_string(region.second) + "])";
        printMatrixSection(result, reference, rows, cols, 
                          region.first, region.second, title, config);
        regions_shown++;
    }
    
    // Show bottom-right corner if different from top-left
    if (rows > config.max_rows || cols > config.max_cols) {
        int start_row = std::max(0, rows - config.max_rows);
        int start_col = std::max(0, cols - config.max_cols);
        printMatrixSection(result, reference, rows, cols, start_row, start_col,
                          "Bottom-Right Corner", config);
    }
    
    std::cout << "\n" << Colors::DIM << "Note: Use smaller matrices or adjust max_rows/max_cols for complete view" 
              << Colors::RESET << std::endl;
}

/**
 * @brief Pretty print a single matrix (simple interface)
 * 
 * @param data Matrix data
 * @param rows Logical number of rows
 * @param cols Logical number of columns
 * @param name Matrix name (optional)
 * @param storage_format 'R' for row-major (default), 'C' for column-major
 * @param max_display Maximum rows/cols to display (default 16)
 */
template<typename T>
void printMatrix(const T* data, int rows, int cols, 
                const std::string& name = "", 
                char storage_format = 'R',
                int max_display = 16) {
    
    // Print header
    if (!name.empty()) {
        std::cout << "\n" << Colors::BOLD << Colors::CYAN 
                  << "Matrix " << name << " (" << rows << " × " << cols << ", "
                  << (storage_format == 'R' ? "row-major" : "column-major") << ")" 
                  << Colors::RESET << std::endl;
    } else {
        std::cout << "\n" << Colors::BOLD << Colors::CYAN 
                  << "Matrix (" << rows << " × " << cols << ", "
                  << (storage_format == 'R' ? "row-major" : "column-major") << ")" 
                  << Colors::RESET << std::endl;
    }
    
    int display_rows = std::min(rows, max_display);
    int display_cols = std::min(cols, max_display);
    
    if (rows > max_display || cols > max_display) {
        std::cout << Colors::DIM << "Showing top-left " << display_rows 
                  << "×" << display_cols << " region" << Colors::RESET << std::endl;
    }
    
    // Print column indices
    std::cout << "    ";
    for (int j = 0; j < display_cols; ++j) {
        std::cout << std::setw(8) << j;
    }
    std::cout << std::endl;
    
    // Print separator
    std::cout << "    ";
    for (int j = 0; j < display_cols; ++j) {
        std::cout << "--------";
    }
    std::cout << std::endl;
    
    // Print matrix data using logical coordinates
    for (int i = 0; i < display_rows; ++i) {
        std::cout << std::setw(3) << i << ":";
        
        for (int j = 0; j < display_cols; ++j) {
            // Calculate physical index based on storage format
            int idx;
            if (storage_format == 'R') {
                // Row-major: data[i][j] = data[i * cols + j]
                idx = i * cols + j;
            } else {
                // Column-major: data[i][j] = data[j * rows + i]
                idx = j * rows + i;
            }
            
            float val = toFloat(data[idx]);
            std::cout << std::fixed << std::setprecision(1) 
                     << std::setw(8) << val;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * @brief Pretty print a single matrix (vector interface)
 */
template<typename T>
void printMatrix(const std::vector<T>& data, int rows, int cols, 
                const std::string& name = "", 
                char storage_format = 'R',
                int max_display = 16) {
    printMatrix(data.data(), rows, cols, name, storage_format, max_display);
}

/**
 * @brief Print input matrices for reference (with storage format support)
 * 
 * @param A Matrix A data
 * @param B Matrix B data
 * @param M Number of rows in A (logical)
 * @param K Number of columns in A / rows in B (logical)
 * @param N Number of columns in B (logical)
 * @param lhs_format Storage format of A ('R' or 'C')
 * @param rhs_format Storage format of B ('R' or 'C')
 * @param config Print configuration
 */
template<typename T>
void printInputMatrices(const std::vector<T>& A, const std::vector<T>& B,
                       int M, int K, int N, 
                       char lhs_format = 'R', char rhs_format = 'R',
                       const MatrixPrintConfig& config = MatrixPrintConfig{}) {
    
    std::cout << "\n" << Colors::BOLD << Colors::BLUE 
              << "═══════════════════════════════════════════════════════════════" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << Colors::BLUE << "  Input Matrices" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << Colors::BLUE 
              << "═══════════════════════════════════════════════════════════════" << Colors::RESET << std::endl;
    
    // Print matrix A
    printMatrix(A, M, K, "A", lhs_format, config);
    
    // Print matrix B
    printMatrix(B, K, N, "B", rhs_format, config);
}

/**
 * @brief Print input matrices for reference (backward compatibility)
 */
template<typename T>
void printInputMatrices(const std::vector<T>& A, const std::vector<T>& B,
                       int M, int K, int N, const MatrixPrintConfig& config = MatrixPrintConfig{}) {
    
    std::cout << "\n" << Colors::BOLD << Colors::BLUE 
              << "═══════════════════════════════════════════════════════════════" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << Colors::BLUE << "  Input Matrices" << Colors::RESET << std::endl;
    std::cout << Colors::BOLD << Colors::BLUE 
              << "═══════════════════════════════════════════════════════════════" << Colors::RESET << std::endl;
    
    // Print matrix A (M × K)
    std::cout << "\n" << Colors::BOLD << "Matrix A (" << M << " × " << K << "):" << Colors::RESET << std::endl;
    if (M <= config.max_rows && K <= config.max_cols) {
        printMatrixHeader(K, 0, config);
        for (int i = 0; i < M; ++i) {
            if (config.show_indices) {
                std::cout << std::setw(3) << i << ":";
            }
            for (int j = 0; j < K; ++j) {
                float val = toFloat(A[i * K + j]);
                std::cout << std::fixed << std::setprecision(config.precision) 
                         << std::setw(config.width) << val;
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Matrix too large, showing top-left " << config.max_rows 
                  << "×" << config.max_cols << " region:" << std::endl;
        printMatrixHeader(std::min(K, config.max_cols), 0, config);
        for (int i = 0; i < std::min(M, config.max_rows); ++i) {
            if (config.show_indices) {
                std::cout << std::setw(3) << i << ":";
            }
            for (int j = 0; j < std::min(K, config.max_cols); ++j) {
                float val = toFloat(A[i * K + j]);
                std::cout << std::fixed << std::setprecision(config.precision) 
                         << std::setw(config.width) << val;
            }
            std::cout << std::endl;
        }
    }
    
    // Print matrix B (K × N)
    std::cout << "\n" << Colors::BOLD << "Matrix B (" << K << " × " << N << "):" << Colors::RESET << std::endl;
    if (K <= config.max_rows && N <= config.max_cols) {
        printMatrixHeader(N, 0, config);
        for (int i = 0; i < K; ++i) {
            if (config.show_indices) {
                std::cout << std::setw(3) << i << ":";
            }
            for (int j = 0; j < N; ++j) {
                float val = toFloat(B[i * N + j]);
                std::cout << std::fixed << std::setprecision(config.precision) 
                         << std::setw(config.width) << val;
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Matrix too large, showing top-left " << config.max_rows 
                  << "×" << config.max_cols << " region:" << std::endl;
        printMatrixHeader(std::min(N, config.max_cols), 0, config);
        for (int i = 0; i < std::min(K, config.max_rows); ++i) {
            if (config.show_indices) {
                std::cout << std::setw(3) << i << ":";
            }
            for (int j = 0; j < std::min(N, config.max_cols); ++j) {
                float val = toFloat(B[i * N + j]);
                std::cout << std::fixed << std::setprecision(config.precision) 
                         << std::setw(config.width) << val;
            }
            std::cout << std::endl;
        }
    }
}

} // namespace utils
} // namespace cutekernellib