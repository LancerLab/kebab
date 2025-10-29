#include "cutekernellib/config/config_parser.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace cutekernellib {
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

} // namespace config
} // namespace cutekernellib
