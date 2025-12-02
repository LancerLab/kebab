/**
 * @file data_size.h
 * @brief Data size formatting and parsing utilities
 *
 * Provides utilities for:
 * - Formatting byte sizes to human-readable strings (KB, MB, GB)
 * - Parsing data size strings to bytes
 * - Common data size constants
 */

#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <cstddef>
#include <stdexcept>

namespace kebab {
namespace utils {

// ============================================================================
// Data Size Constants
// ============================================================================

constexpr size_t KB = 1024ULL;
constexpr size_t MB = 1024ULL * KB;
constexpr size_t GB = 1024ULL * MB;
constexpr size_t TB = 1024ULL * GB;

// Common sizes for benchmarks
constexpr size_t SIZE_1KB   = 1 * KB;
constexpr size_t SIZE_4KB   = 4 * KB;
constexpr size_t SIZE_16KB  = 16 * KB;
constexpr size_t SIZE_32KB  = 32 * KB;
constexpr size_t SIZE_64KB  = 64 * KB;
constexpr size_t SIZE_128KB = 128 * KB;
constexpr size_t SIZE_256KB = 256 * KB;
constexpr size_t SIZE_512KB = 512 * KB;
constexpr size_t SIZE_1MB   = 1 * MB;
constexpr size_t SIZE_2MB   = 2 * MB;
constexpr size_t SIZE_4MB   = 4 * MB;
constexpr size_t SIZE_8MB   = 8 * MB;
constexpr size_t SIZE_16MB  = 16 * MB;
constexpr size_t SIZE_32MB  = 32 * MB;
constexpr size_t SIZE_64MB  = 64 * MB;
constexpr size_t SIZE_128MB = 128 * MB;
constexpr size_t SIZE_256MB = 256 * MB;
constexpr size_t SIZE_512MB = 512 * MB;
constexpr size_t SIZE_1GB   = 1 * GB;

// ============================================================================
// Formatting Functions
// ============================================================================

/**
 * @brief Format byte size to human-readable string
 *
 * @param bytes Size in bytes
 * @param precision Decimal precision (default: 1)
 * @return Human-readable size string (e.g., "4.0 MB", "512 KB")
 */
inline std::string formatBytes(size_t bytes, int precision = 1) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision);

    if (bytes >= GB) {
        ss << static_cast<double>(bytes) / GB << " GB";
    } else if (bytes >= MB) {
        ss << static_cast<double>(bytes) / MB << " MB";
    } else if (bytes >= KB) {
        ss << static_cast<double>(bytes) / KB << " KB";
    } else {
        ss << bytes << " B";
    }
    return ss.str();
}

/**
 * @brief Format byte size to compact string (no space)
 *
 * @param bytes Size in bytes
 * @return Compact size string (e.g., "4MB", "512KB")
 */
inline std::string formatBytesCompact(size_t bytes) {
    std::ostringstream ss;

    if (bytes >= GB && bytes % GB == 0) {
        ss << (bytes / GB) << "GB";
    } else if (bytes >= MB && bytes % MB == 0) {
        ss << (bytes / MB) << "MB";
    } else if (bytes >= KB && bytes % KB == 0) {
        ss << (bytes / KB) << "KB";
    } else {
        ss << bytes << "B";
    }
    return ss.str();
}

/**
 * @brief Parse data size string to bytes
 *
 * Supports formats: "4MB", "4 MB", "4mb", "4096KB", "4096", etc.
 *
 * @param size_str Size string to parse
 * @return Size in bytes
 * @throws std::invalid_argument if format is invalid
 */
inline size_t parseBytes(const std::string& size_str) {
    std::string s = size_str;
    // Remove spaces
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    
    // Convert to uppercase for comparison
    std::string upper = s;
    for (auto& c : upper) c = std::toupper(c);

    size_t multiplier = 1;
    size_t suffix_len = 0;

    if (upper.length() >= 2) {
        std::string suffix = upper.substr(upper.length() - 2);
        if (suffix == "TB") { multiplier = TB; suffix_len = 2; }
        else if (suffix == "GB") { multiplier = GB; suffix_len = 2; }
        else if (suffix == "MB") { multiplier = MB; suffix_len = 2; }
        else if (suffix == "KB") { multiplier = KB; suffix_len = 2; }
    }
    if (suffix_len == 0 && upper.length() >= 1) {
        char last = upper.back();
        if (last == 'T') { multiplier = TB; suffix_len = 1; }
        else if (last == 'G') { multiplier = GB; suffix_len = 1; }
        else if (last == 'M') { multiplier = MB; suffix_len = 1; }
        else if (last == 'K') { multiplier = KB; suffix_len = 1; }
        else if (last == 'B') { suffix_len = 1; }
    }

    std::string num_str = s.substr(0, s.length() - suffix_len);
    if (num_str.empty()) {
        throw std::invalid_argument("Invalid size string: " + size_str);
    }

    double value = std::stod(num_str);
    return static_cast<size_t>(value * multiplier);
}

/**
 * @brief Calculate number of elements from byte size and element type
 */
template<typename T>
inline size_t bytesToElements(size_t bytes) {
    return bytes / sizeof(T);
}

/**
 * @brief Calculate byte size from number of elements and element type
 */
template<typename T>
inline size_t elementsToBytes(size_t elements) {
    return elements * sizeof(T);
}

} // namespace utils
} // namespace kebab

