#include "kebab/config/config_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace kebab {
namespace config {

// Implementation class to hide yaml-cpp dependency from header
class ConfigParser::Impl {
public:
    YAML::Node config;
    std::string config_path;
    static int next_id;
    int id;
    
    Impl() : id(next_id++) {}
    
    ~Impl() {}
    
    void load(const std::string& path) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Configuration file not found: " + path);
        }
        
        try {
            config_path = path;
            config = YAML::LoadFile(path);
        } catch (const YAML::Exception& e) {
            throw std::runtime_error(
                "Failed to parse YAML configuration file: " + 
                std::string(e.what())
            );
        }
    }
    
    YAML::Node getConfig() {
        // Workaround: reload config if it's not defined
        // This handles an issue with YAML::Node becoming undefined in certain contexts
        if (!config.IsDefined() && !config_path.empty()) {
            config = YAML::LoadFile(config_path);
        }
        // Always return a fresh copy to avoid node invalidation issues
        if (!config_path.empty()) {
            return YAML::LoadFile(config_path);
        }
        return config;
    }
    
    template<typename T>
    T get(const std::string& key, const T& default_value) {
        try {
            std::vector<std::string> keys;
            size_t start = 0;
            size_t end = key.find('.');
            
            while (end != std::string::npos) {
                keys.push_back(key.substr(start, end - start));
                start = end + 1;
                end = key.find('.', start);
            }
            keys.push_back(key.substr(start));
            
            YAML::Node node = getConfig();
            for (const auto& k : keys) {
                node = node[k];
                if (!node.IsDefined()) {
                    return default_value;
                }
            }
            
            return node.as<T>();
        } catch (const YAML::Exception&) {
            return default_value;
        }
    }
    
    template<typename T>
    std::vector<T> getVector(const std::string& key, const std::vector<T>& default_value) {
        try {
            std::vector<std::string> keys;
            size_t start = 0;
            size_t end = key.find('.');
            
            while (end != std::string::npos) {
                keys.push_back(key.substr(start, end - start));
                start = end + 1;
                end = key.find('.', start);
            }
            keys.push_back(key.substr(start));
            
            YAML::Node node = getConfig();
            for (const auto& k : keys) {
                node = node[k];
                if (!node.IsDefined()) {
                    return default_value;
                }
            }
            
            if (!node.IsSequence()) {
                return default_value;
            }
            
            return node.as<std::vector<T>>();
        } catch (const YAML::Exception&) {
            return default_value;
        }
    }
};

int ConfigParser::Impl::next_id = 0;

ConfigParser& ConfigParser::getInstance(const std::string& config_path) {
    static ConfigParser instance;
    
    if (!instance.loaded_ || (!config_path.empty() && config_path != instance.config_path_)) {
        instance.load(config_path);
    }
    
    return instance;
}

void ConfigParser::load(const std::string& config_path) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    
    try {
        impl_->load(config_path);
        config_path_ = config_path;
        loaded_ = true;
        validateConfig();
        
        std::cout << "Configuration loaded successfully from: " << config_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to load configuration" << std::endl;
        std::cerr << "  File: " << config_path << std::endl;
        std::cerr << "  Error: " << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << "Please ensure:" << std::endl;
        std::cerr << "  1. config.yaml exists in the current directory" << std::endl;
        std::cerr << "  2. The file contains valid YAML syntax" << std::endl;
        std::cerr << "  3. All required fields are present" << std::endl;
        throw;
    }
}

void ConfigParser::validateConfig() {
    // Validate required fields exist
    if (!impl_->config["build"]) {
        throw std::runtime_error("Missing required 'build' section in config.yaml");
    }
    if (!impl_->config["benchmark"]) {
        throw std::runtime_error("Missing required 'benchmark' section in config.yaml");
    }
    
    // Validate build mode
    std::string mode = getBuildMode();
    if (mode != "debug" && mode != "release") {
        std::cerr << "WARNING: Invalid build mode '" << mode 
                  << "'. Using 'release' as default." << std::endl;
    }
    
    // Validate warmup and measurement runs
    if (getWarmupRuns() < 0) {
        throw std::runtime_error("warmup_runs must be non-negative");
    }
    if (getMeasurementRuns() <= 0) {
        throw std::runtime_error("measurement_runs must be positive");
    }
}

void ConfigParser::reload(const std::string& config_path) {
    std::string path = config_path.empty() ? config_path_ : config_path;
    loaded_ = false;
    load(path);
}

// Build configuration
std::string ConfigParser::getBuildMode() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->get<std::string>("build.mode", "release");
}

std::string ConfigParser::getOptimizationLevel() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->get<std::string>("build.optimization", "O3");
}

std::string ConfigParser::getCudaArch() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->get<std::string>("build.cuda_arch", "auto");
}

// Benchmark configuration
int ConfigParser::getWarmupRuns() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->get<int>("benchmark.warmup_runs", 10);
}

int ConfigParser::getMeasurementRuns() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->get<int>("benchmark.measurement_runs", 100);
}

std::vector<int> ConfigParser::getBatchSizes() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->getVector<int>("benchmark.batch_sizes", {256, 512, 1024, 2048, 4096});
}

std::vector<std::string> ConfigParser::getDataTypes() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->getVector<std::string>("benchmark.data_types", {"float32"});
}

// Profiling configuration
std::vector<std::string> ConfigParser::getProfilingMetrics() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->getVector<std::string>("profiling.ncu_metrics", {});
}

std::vector<std::string> ConfigParser::getProfilingSections() const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    return impl_->getVector<std::string>("profiling.sections", {"SpeedOfLight"});
}

// Operator configuration
bool ConfigParser::isOperatorEnabled(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".enabled";
    return impl_->get<bool>(key, false);
}

std::vector<int> ConfigParser::getOperatorSizes(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".sizes";
    return impl_->getVector<int>(key, {});
}

std::vector<int> ConfigParser::getOperatorTileSizes(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".tile_sizes";
    return impl_->getVector<int>(key, {});
}

std::vector<int> ConfigParser::getOperatorMatrixSizes(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".matrix_sizes";
    return impl_->getVector<int>(key, {});
}

std::vector<std::string> ConfigParser::getOperatorModes(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".modes";
    return impl_->getVector<std::string>(key, {"NT"});
}

std::vector<std::string> ConfigParser::getOperatorPrecisions(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".precisions";
    return impl_->getVector<std::string>(key, {"float16"});
}

std::string ConfigParser::getOperatorImpl(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    // Try vector first (new format)
    auto impls = getOperatorImpls(op_name);
    if (!impls.empty()) {
        return impls[0];
    }
    // Fall back to single value (old format)
    std::string key = "operators." + op_name + ".impl";
    return impl_->get<std::string>(key, "cute");
}

std::vector<std::string> ConfigParser::getOperatorImpls(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".impl";

    // First try to get as vector
    auto result = impl_->getVector<std::string>(key, {});
    if (!result.empty()) {
        return result;
    }

    // Fall back to single value
    std::string single = impl_->get<std::string>(key, "");
    if (!single.empty()) {
        return {single};
    }

    // Default
    return {"cute"};
}

int ConfigParser::getOperatorVersion(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    // Try vector first (new format: "versions")
    auto versions = getOperatorVersions(op_name);
    if (!versions.empty()) {
        return versions[0];
    }
    // Fall back to single value (old format: "version")
    std::string key = "operators." + op_name + ".version";
    return impl_->get<int>(key, 1);
}

std::vector<int> ConfigParser::getOperatorVersions(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }

    // First try "versions" (new format, plural)
    std::string key_plural = "operators." + op_name + ".versions";
    auto result = impl_->getVector<int>(key_plural, {});
    if (!result.empty()) {
        return result;
    }

    // Fall back to "version" (old format, singular)
    std::string key_singular = "operators." + op_name + ".version";
    int single = impl_->get<int>(key_singular, -1);
    if (single >= 0) {
        return {single};
    }

    // Default
    return {1};
}

std::string ConfigParser::getOperatorInitMethod(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".init_method";
    return impl_->get<std::string>(key, "rand-rand");
}

bool ConfigParser::getOperatorVerbose(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".verbose";
    return impl_->get<bool>(key, false);
}

int ConfigParser::getOperatorGpuId(const std::string& op_name) const {
    if (!loaded_) {
        throw std::runtime_error("Configuration not loaded");
    }
    std::string key = "operators." + op_name + ".gpu_id";
    return impl_->get<int>(key, 0);  // Default to GPU 0
}



} // namespace config
} // namespace kebab
