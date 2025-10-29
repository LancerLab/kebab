# CuTeKernelLib Makefile
# High-performance kernel library using NVIDIA CUTLASS CuTe

.PHONY: all setup build clean test help

# ============================================================================
# OS Detection
# ============================================================================
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Linux)
    OS := Linux
    RM := rm -rf
    MKDIR := mkdir -p
else ifeq ($(UNAME_S),Darwin)
    OS := MacOS
    RM := rm -rf
    MKDIR := mkdir -p
else
    # Assume Windows if uname fails
    OS := Windows
    RM := del /Q /S
    MKDIR := mkdir
endif

# ============================================================================
# GPU Architecture Detection
# ============================================================================
# Allow manual override via environment variable
ifdef CUDA_ARCH
    # User specified CUDA_ARCH manually
    $(info Using manually specified CUDA_ARCH: $(CUDA_ARCH))
else
    # Detect GPU compute capability using nvidia-smi
    COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
    
    # Convert compute capability to CUDA architecture (e.g., 8.0 -> sm_80)
    ifneq ($(COMPUTE_CAP),)
        CUDA_ARCH := sm_$(COMPUTE_CAP)
    else
        # nvidia-smi failed, try alternative detection using deviceQuery if available
        DEVICE_QUERY := $(shell which deviceQuery 2>/dev/null)
        ifneq ($(DEVICE_QUERY),)
            COMPUTE_CAP := $(shell deviceQuery | grep "CUDA Capability" | head -n1 | awk '{print $$5}' | tr -d '.')
            ifneq ($(COMPUTE_CAP),)
                CUDA_ARCH := sm_$(COMPUTE_CAP)
            endif
        endif
        
        # If still not detected, provide helpful error message
        ifndef CUDA_ARCH
            $(warning ========================================)
            $(warning GPU Detection Failed!)
            $(warning ========================================)
            $(warning Could not detect GPU compute capability.)
            $(warning Please ensure:)
            $(warning   1. NVIDIA GPU drivers are installed)
            $(warning   2. nvidia-smi is in your PATH)
            $(warning   3. GPU is visible to the system)
            $(warning )
            $(warning Run 'nvidia-smi' manually to verify GPU visibility.)
            $(warning )
            $(warning If you want to specify architecture manually,)
            $(warning set CUDA_ARCH environment variable:)
            $(warning   export CUDA_ARCH=sm_80)
            $(warning   make build)
            $(warning )
            $(warning Common architectures:)
            $(warning   sm_80 - Ampere (A100, RTX 30xx))
            $(warning   sm_86 - Ampere (RTX 30xx mobile))
            $(warning   sm_89 - Ada Lovelace (RTX 40xx))
            $(warning   sm_90 - Hopper (H100))
            $(warning ========================================)
            $(error GPU detection failed. Cannot proceed without valid CUDA_ARCH.)
        endif
    endif
endif

# ============================================================================
# Compiler and Flags
# ============================================================================
NVCC := nvcc
CXX := g++

# Base NVCC flags with detected architecture
NVCC_FLAGS := -std=c++17 \
              -arch=$(CUDA_ARCH) \
              -I./include \
              -I./third_party/cute/include \
              --expt-relaxed-constexpr \
              --expt-extended-lambda

# Build mode specific flags (will be set from config.yaml)
BUILD_MODE ?= release
ifeq ($(BUILD_MODE),debug)
    NVCC_FLAGS += -g -G -O0 -lineinfo
    CXX_FLAGS := -g -O0
else
    NVCC_FLAGS += -O3 --use_fast_math -lineinfo
    CXX_FLAGS := -O3 -march=native
endif

# Additional flags
NVCC_FLAGS += -Xcompiler -fPIC
CXX_FLAGS += -std=c++17 -fPIC -I./include

# ============================================================================
# Directories
# ============================================================================
BUILD_DIR := build
SRC_DIR := src
INCLUDE_DIR := include
THIRD_PARTY_DIR := third_party
CUTE_DIR := $(THIRD_PARTY_DIR)/cute
BASELINES_DIR := baselines
BENCHMARKS_DIR := benchmarks
BENCH_RESULTS_DIR := bench_results
PROFILING_DIR := profiling

# ============================================================================
# Targets
# ============================================================================

all: help

help:
	@echo "=========================================="
	@echo "CuTeKernelLib Build System"
	@echo "=========================================="
	@echo "Detected Configuration:"
	@echo "  OS:           $(OS)"
	@echo "  CUDA_ARCH:    $(CUDA_ARCH)"
	@echo "  BUILD_MODE:   $(BUILD_MODE)"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup              - Install dependencies (CuTe, yaml-cpp)"
	@echo "  make build              - Compile library and operators"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make test               - Run unit tests"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Benchmark targets (after build):"
	@echo "  make bench-elementwise_add  - Benchmark element-wise add"
	@echo "  make bench-gemm             - Benchmark GEMM"
	@echo "  make bench-all              - Run all benchmarks"
	@echo ""
	@echo "Profiling targets (after build):"
	@echo "  make tune-elementwise_add   - Profile element-wise add"
	@echo "  make tune-gemm              - Profile GEMM"
	@echo "  make tune-all               - Profile all operators"
	@echo "=========================================="

# Display detected GPU info
gpu-info:
	@echo "=========================================="
	@echo "GPU Information"
	@echo "=========================================="
	@nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
	@echo ""
	@echo "Detected CUDA Architecture: $(CUDA_ARCH)"
	@echo "=========================================="

# ============================================================================
# Setup Target - Install Dependencies
# ============================================================================
YAML_CPP_DIR := $(THIRD_PARTY_DIR)/yaml-cpp

setup:
	@echo "=========================================="
	@echo "Setting up CuTeKernelLib dependencies..."
	@echo "=========================================="
	@echo ""
	@echo "[1/4] Creating directory structure..."
	@$(MKDIR) $(THIRD_PARTY_DIR)
	@$(MKDIR) $(BUILD_DIR)
	@$(MKDIR) $(BENCH_RESULTS_DIR)
	@$(MKDIR) $(PROFILING_DIR)
	@echo "  ✓ Directories created"
	@echo ""
	@echo "[2/4] Installing CuTe (CUTLASS)..."
	@if [ -d "$(CUTE_DIR)" ]; then \
		echo "  ℹ CuTe already exists at $(CUTE_DIR)"; \
		echo "  Updating to latest version..."; \
		cd $(CUTE_DIR) && git pull || echo "  ⚠ Could not update (using existing version)"; \
	else \
		echo "  Cloning CUTLASS repository..."; \
		git clone --depth 1 https://github.com/NVIDIA/cutlass.git $(CUTE_DIR) || \
		{ echo ""; \
		  echo "  ✗ ERROR: Failed to clone CuTe repository"; \
		  echo ""; \
		  echo "  Troubleshooting:"; \
		  echo "    1. Check internet connection"; \
		  echo "    2. Verify git is installed: git --version"; \
		  echo "    3. Try manual clone:"; \
		  echo "       git clone https://github.com/NVIDIA/cutlass.git $(CUTE_DIR)"; \
		  echo ""; \
		  exit 1; \
		}; \
	fi
	@echo "  ✓ CuTe installed at $(CUTE_DIR)"
	@echo ""
	@echo "[3/4] Initializing git submodules..."
	@cd $(CUTE_DIR) && git submodule update --init --recursive || \
		{ echo "  ⚠ Warning: Could not initialize submodules (may not be critical)"; }
	@echo "  ✓ Submodules initialized"
	@echo ""
	@echo "[4/4] Installing yaml-cpp..."
	@if [ -d "$(YAML_CPP_DIR)" ]; then \
		echo "  ℹ yaml-cpp already exists at $(YAML_CPP_DIR)"; \
	else \
		echo "  Cloning yaml-cpp repository..."; \
		git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp.git $(YAML_CPP_DIR) || \
		{ echo ""; \
		  echo "  ✗ ERROR: Failed to clone yaml-cpp repository"; \
		  echo ""; \
		  echo "  Troubleshooting:"; \
		  echo "    1. Check internet connection"; \
		  echo "    2. Try manual clone:"; \
		  echo "       git clone https://github.com/jbeder/yaml-cpp.git $(YAML_CPP_DIR)"; \
		  echo ""; \
		  exit 1; \
		}; \
	fi
	@echo "  Building yaml-cpp..."
	@cd $(YAML_CPP_DIR) && \
		$(MKDIR) build && \
		cd build && \
		cmake -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
		make -j$$(nproc) || \
		{ echo "  ✗ ERROR: Failed to build yaml-cpp"; \
		  echo "  Ensure cmake is installed: sudo apt-get install cmake"; \
		  exit 1; \
		}
	@echo "  ✓ yaml-cpp installed and built"
	@echo ""
	@echo "=========================================="
	@echo "Setup complete!"
	@echo "=========================================="
	@echo ""
	@echo "Dependencies installed:"
	@echo "  - CuTe headers:  $(CUTE_DIR)/include"
	@echo "  - yaml-cpp:      $(YAML_CPP_DIR)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make build' to compile the library"
	@echo "  2. Run 'make bench-<operator>' to benchmark operators"
	@echo "=========================================="

clean:
	@echo "Cleaning build artifacts..."
	$(RM) $(BUILD_DIR)
	$(RM) $(BENCH_RESULTS_DIR)/*.csv
	$(RM) $(PROFILING_DIR)/*.ncu-rep
	$(RM) $(PROFILING_DIR)/*.txt
	@echo "Clean complete."
