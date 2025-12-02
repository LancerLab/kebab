#include "microbench/microbench.h"
#include <algorithm>
#include <cmath>

namespace kebab {
namespace microbench {

void MicrobenchReport::printTable() const {
    if (results_.empty()) {
        std::cout << "No results to display.\n";
        return;
    }
    
    // Find max widths for formatting
    size_t max_name_len = 20;
    size_t max_desc_len = 30;
    for (const auto& r : results_) {
        max_name_len = std::max(max_name_len, r.variant_name.length());
        max_desc_len = std::max(max_desc_len, r.description.length());
    }
    max_name_len = std::min(max_name_len, size_t(25));
    max_desc_len = std::min(max_desc_len, size_t(35));
    
    // Get unique data sizes for grouping
    std::vector<size_t> data_sizes;
    for (const auto& r : results_) {
        if (std::find(data_sizes.begin(), data_sizes.end(), r.data_size_bytes) == data_sizes.end()) {
            data_sizes.push_back(r.data_size_bytes);
        }
    }
    std::sort(data_sizes.begin(), data_sizes.end());
    
    // Print results grouped by data size
    for (size_t sz : data_sizes) {
        std::cout << "\n";
        std::cout << "Data Size: " << formatBytes(sz) << "\n";
        std::cout << std::string(100, '-') << "\n";
        
        // Header
        std::cout << std::left
                  << std::setw(max_name_len + 2) << "Variant"
                  << std::setw(14) << "Latency (us)"
                  << std::setw(14) << "BW (GB/s)"
                  << std::setw(12) << "Efficiency"
                  << std::setw(10) << "Speedup"
                  << std::endl;
        std::cout << std::string(100, '-') << "\n";
        
        // Find baseline for this size
        float baseline_latency = 0.0f;
        for (const auto& r : results_) {
            if (r.data_size_bytes == sz && r.is_baseline) {
                baseline_latency = r.latency_us;
                break;
            }
        }
        if (baseline_latency <= 0.0f) {
            // Use first result as baseline if none marked
            for (const auto& r : results_) {
                if (r.data_size_bytes == sz) {
                    baseline_latency = r.latency_us;
                    break;
                }
            }
        }
        
        // Print results for this size
        for (const auto& r : results_) {
            if (r.data_size_bytes != sz) continue;
            
            float speedup = (r.latency_us > 0.0f && baseline_latency > 0.0f) 
                          ? baseline_latency / r.latency_us : 1.0f;
            
            std::string eff_str = std::to_string(static_cast<int>(r.efficiency_pct)) + "%";
            std::string speedup_str = (speedup >= 0.995f && speedup <= 1.005f) 
                                    ? "1.00x (base)" 
                                    : (std::to_string(speedup).substr(0, 4) + "x");
            
            std::cout << std::left
                      << std::setw(max_name_len + 2) << r.variant_name
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.latency_us
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.bandwidth_gbps
                      << std::setw(12) << eff_str
                      << std::setw(10) << speedup_str
                      << std::endl;
        }
    }
}

void MicrobenchReport::printSummary() const {
    std::cout << "\n";
    std::cout << std::string(100, '=') << "\n";
    std::cout << "Summary:\n";
    
    // Find best variant for each size
    std::vector<size_t> data_sizes;
    for (const auto& r : results_) {
        if (std::find(data_sizes.begin(), data_sizes.end(), r.data_size_bytes) == data_sizes.end()) {
            data_sizes.push_back(r.data_size_bytes);
        }
    }
    
    for (size_t sz : data_sizes) {
        float best_bw = 0.0f;
        std::string best_variant;
        for (const auto& r : results_) {
            if (r.data_size_bytes == sz && r.bandwidth_gbps > best_bw) {
                best_bw = r.bandwidth_gbps;
                best_variant = r.variant_name;
            }
        }
        std::cout << "  " << formatBytes(sz) << ": Best = " << best_variant 
                  << " (" << std::fixed << std::setprecision(2) << best_bw << " GB/s)\n";
    }
    std::cout << std::string(100, '=') << "\n";
}

std::string MicrobenchReport::formatBytes(size_t bytes) const {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " B";
}

} // namespace microbench
} // namespace kebab

