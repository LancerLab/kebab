#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace cutekernellib {
namespace config {

/**
 * @brief Singleton class for parsing and accessing configuration from config.yaml
 * 
 * This class provides centralized access to all configuration parameters
 * including build settings, benchmark parameters, and operator configurations.
 */
class ConfigParser {
public:
    /**
     * @brief Get singleton instance of ConfigParser
     * @param config_path Path to config.yaml file (default: "config.yaml")
     * @return Reference to singleton instance
     */
    static ConfigParser& getInstance(const std::string& config_path = "config.yaml");
    
    // Prevent copying
    ConfigParser(const ConfigParser&) = delete;
    ConfigParser& operator=(const ConfigParser&) = delete;
    
    // Build configuration
    std::string getBuildMode() const;
    std::string getOptimizationLevel() const;
    std::string getCudaArch() const;
    
    // Benchmark configuration
    int getWarmupRuns() const;
    int getMeasurementRuns() const;
    std::vector<int> getBatchSizes() const;
    std::vector<std::string> getDataTypes() const;
    
    // Profiling configuration
    std::vector<std::string> getProfilingMetrics() const;
    std::vector<std::string> getProfilingSections() const;
    
    // Operator configuration
    bool isOperatorEnabled(const std::string& op_name) const;
    std::vector<int> getOperatorSizes(const std::string& op_name) const;
    std::vector<int> getOperatorTileSizes(const std::string& op_name) const;
    std::vector<int> getOperatorMatrixSizes(const std::string& op_name) const;
    
    // Utility
    void reload(const std::string& config_path = "");
    std::string getConfigPath() const { return config_path_; }
    
private:
    ConfigParser() = default;
    ~ConfigParser() = default;
    
    void load(const std::string& config_path);
    void validateConfig();
    
    // Forward declaration of implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::string config_path_;
    bool loaded_ = false;
};

} // namespace config
} // namespace cutekernellib
