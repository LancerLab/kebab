# CuTeKernelLib Makefile
# High-performance kernel library using NVIDIA CUTLASS CuTe

.PHONY: all setup build clean help test

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

# Convert sm_90 to sm_90a for WGMMA support
ifeq ($(CUDA_ARCH),sm_90)
    CUDA_ARCH := sm_90a
    $(info Converting sm_90 to sm_90a for WGMMA support)
endif

# ============================================================================
# CUDA Toolkit Detection with Driver Compatibility
# ============================================================================
# Allow manual override via environment variable
ifndef CUDA_PATH
    # Step 1: Detect driver-supported CUDA version
    # Based on NVIDIA official compatibility matrix
    DRIVER_CUDA_VERSION := $(shell nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | \
        awk 'BEGIN{FS="."} { \
            if ($$1 >= 580) print "13.0"; \
            else if ($$1 >= 570) print "12.8"; \
            else if ($$1 >= 560) print "12.6"; \
            else if ($$1 >= 550) print "12.4"; \
            else if ($$1 >= 535) print "12.2"; \
            else if ($$1 >= 525) print "12.0"; \
            else if ($$1 >= 515) print "11.8"; \
            else if ($$1 >= 510) print "11.6"; \
            else if ($$1 >= 495) print "11.4"; \
            else if ($$1 >= 470) print "11.2"; \
            else if ($$1 >= 460) print "11.0"; \
            else print "10.2"; \
        }')
    
    # Step 2: Find all available CUDA installations
    AVAILABLE_CUDA_PATHS := $(wildcard /usr/local/cuda-* /opt/cuda-*)
    AVAILABLE_CUDA_PATHS += $(wildcard /usr/local/cuda)
    
    # Step 3: Extract version numbers and find compatible versions
    define extract_cuda_version
        $(shell echo $(1) | sed -n 's|.*/cuda-\([0-9]\+\.[0-9]\+\).*|\1|p')
    endef
    
    # Step 4: Find the best compatible CUDA version
    BEST_CUDA_PATH := 
    BEST_CUDA_VERSION := 0.0
    
    # Check if we detected driver version
    ifneq ($(DRIVER_CUDA_VERSION),)
        $(info Detected driver supports CUDA version: $(DRIVER_CUDA_VERSION))
        
        # Convert driver version to comparable number (e.g., 12.8 -> 1208)
        DRIVER_VERSION_NUM := $(shell echo $(DRIVER_CUDA_VERSION) | awk -F. '{printf "%d%02d", $$1, $$2}')
        
        # Check each available CUDA installation
        $(foreach path,$(AVAILABLE_CUDA_PATHS), \
            $(eval CUDA_VER := $(call extract_cuda_version,$(path))) \
            $(if $(CUDA_VER), \
                $(eval CUDA_VER_NUM := $(shell echo $(CUDA_VER) | awk -F. '{printf "%d%02d", $$1, $$2}')) \
                $(if $(shell [ $(CUDA_VER_NUM) -le $(DRIVER_VERSION_NUM) ] && [ $(CUDA_VER_NUM) -gt $(shell echo $(BEST_CUDA_VERSION) | awk -F. '{printf "%d%02d", $$1, $$2}') ] && echo yes), \
                    $(eval BEST_CUDA_PATH := $(path)) \
                    $(eval BEST_CUDA_VERSION := $(CUDA_VER)) \
                ) \
            ) \
        )
        
        # Special handling for /usr/local/cuda symlink
        ifneq ($(wildcard /usr/local/cuda),)
            SYMLINK_TARGET := $(shell readlink -f /usr/local/cuda 2>/dev/null)
            ifneq ($(SYMLINK_TARGET),)
                SYMLINK_VER := $(call extract_cuda_version,$(SYMLINK_TARGET))
                ifneq ($(SYMLINK_VER),)
                    SYMLINK_VER_NUM := $(shell echo $(SYMLINK_VER) | awk -F. '{printf "%d%02d", $$1, $$2}')
                    ifeq ($(shell [ $(SYMLINK_VER_NUM) -le $(DRIVER_VERSION_NUM) ] && [ $(SYMLINK_VER_NUM) -gt $(shell echo $(BEST_CUDA_VERSION) | awk -F. '{printf "%d%02d", $$1, $$2}') ] && echo yes),yes)
                        BEST_CUDA_PATH := /usr/local/cuda
                        BEST_CUDA_VERSION := $(SYMLINK_VER)
                    endif
                endif
            endif
        endif
        
        # Use the best compatible version
        ifneq ($(BEST_CUDA_PATH),)
            CUDA_PATH := $(BEST_CUDA_PATH)
            $(info Selected compatible CUDA toolkit: $(CUDA_PATH) (version $(BEST_CUDA_VERSION)))
        else
            $(warning ========================================)
            $(warning CUDA Version Compatibility Issue!)
            $(warning ========================================)
            $(warning Driver supports CUDA $(DRIVER_CUDA_VERSION), but no compatible toolkit found.)
            $(warning Available CUDA installations:)
            $(foreach path,$(AVAILABLE_CUDA_PATHS),$(warning   $(path)))
            $(warning )
            $(warning Consider:)
            $(warning   1. Installing CUDA $(DRIVER_CUDA_VERSION) or earlier)
            $(warning   2. Upgrading your NVIDIA driver)
            $(warning   3. Manually setting CUDA_PATH environment variable)
            $(warning ========================================)
        endif
    endif
    
    # Fallback: If no compatible version found or driver detection failed
    ifeq ($(CUDA_PATH),)
        $(info Driver detection failed or no compatible CUDA found, using fallback detection...)
        # Try to find CUDA installation directory using traditional method
        CUDA_PATH := $(shell which nvcc 2>/dev/null | sed 's|/bin/nvcc||')
        ifeq ($(CUDA_PATH),)
            # Try common CUDA installation paths
            CUDA_SEARCH_PATHS := /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11 /opt/cuda
            CUDA_PATH := $(firstword $(foreach path,$(CUDA_SEARCH_PATHS),$(wildcard $(path))))
        endif
    endif
endif

# If CUDA_PATH is still empty, show error
ifeq ($(CUDA_PATH),)
    $(warning ========================================)
    $(warning CUDA Toolkit Not Found!)
    $(warning ========================================)
    $(warning Could not locate CUDA installation.)
    $(warning Please ensure CUDA toolkit is installed and either:)
    $(warning   1. Add CUDA bin directory to PATH:)
    $(warning      export PATH=/usr/local/cuda/bin:$$PATH)
    $(warning   2. Set CUDA_PATH environment variable:)
    $(warning      export CUDA_PATH=/usr/local/cuda)
    $(warning )
    $(warning Download CUDA from:)
    $(warning   https://developer.nvidia.com/cuda-downloads)
    $(warning ========================================)
    $(error CUDA toolkit not found. Cannot proceed.)
endif

# ============================================================================
# Compiler and Flags
# ============================================================================
# Set CUDA binaries
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++

# Verify NVCC exists
ifeq ($(wildcard $(NVCC)),)
    $(error NVCC not found at $(NVCC). Please check CUDA installation.)
endif

# Base NVCC flags with detected architecture
NVCC_FLAGS := -std=c++17 \
              -arch=$(CUDA_ARCH) \
              -I./include \
              -I./third_party/cute/include \
              --expt-relaxed-constexpr \
              --expt-extended-lambda \
              -cudart shared

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

# Enable SM90 WGMMA support for Hopper
ifeq ($(CUDA_ARCH),sm_90a)
    NVCC_FLAGS += -D__CUDA_ARCH_FEAT_SM90_ALL
endif
CUDA_INCLUDE := $(CUDA_PATH)/include
CXX_FLAGS += -std=c++17 -fPIC -I./include -I$(YAML_CPP_DIR)/include -I$(CUDA_INCLUDE)

# Linker flags
LDFLAGS := -L$(YAML_CPP_DIR)/build -lyaml-cpp

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
# Operators
# ============================================================================
OPERATORS := elementwise_add gemm

# ============================================================================
# Baselines
# ============================================================================
BASELINES := cuda_elementwise_add cuda_gemm

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
	@echo "  CUDA_PATH:    $(CUDA_PATH)"
	@echo "  NVCC:         $(NVCC)"
	@echo "  CUDA_ARCH:    $(CUDA_ARCH)"
	@echo "  BUILD_MODE:   $(BUILD_MODE)"
	@if [ -n "$(NCU)" ]; then \
		echo "  NCU:          $(NCU)"; \
	else \
		echo "  NCU:          Not found (profiling unavailable)"; \
	fi
	@if [ -n "$(NVIDIA_SMI)" ]; then \
		echo "  NVIDIA_SMI:   $(NVIDIA_SMI)"; \
	else \
		echo "  NVIDIA_SMI:   Not found"; \
	fi
	@echo ""
	@echo "Available targets:"
	@echo "  make setup              - Install dependencies (CuTe, yaml-cpp)"
	@echo "  make build              - Compile library and operators"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Test targets:"
	@echo "  make test                   - Run all tests (uses bench-gemm with verification)"
	@echo ""
	@echo "Benchmark targets (after build):"
	@echo "  make bench-elementwise-add  - Benchmark element-wise add"
	@echo "  make bench-gemm             - Benchmark GEMM"
	@echo "  make bench-all              - Run all benchmarks and generate summary"
	@echo ""
	@echo "Profiling targets (after build):"
	@echo "  make tune-elementwise-add   - Profile element-wise add with ncu"
	@echo "  make tune-gemm              - Profile GEMM with ncu"
	@echo "  make tune-all               - Profile all operators with ncu"
	@echo "=========================================="

# Display detected GPU info
gpu-info:
	@echo "=========================================="
	@echo "GPU Information"
	@echo "=========================================="
	@if [ -n "$(NVIDIA_SMI)" ] && [ -x "$(NVIDIA_SMI)" ]; then \
		$(NVIDIA_SMI) --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader; \
	else \
		echo "  ✗ nvidia-smi not found or not executable"; \
		echo "  Please ensure NVIDIA drivers are installed"; \
	fi
	@echo ""
	@echo "Detected CUDA Architecture: $(CUDA_ARCH)"
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "NVCC: $(NVCC)"
	@if [ -n "$(NCU)" ]; then \
		echo "NCU: $(NCU)"; \
	else \
		echo "NCU: Not found"; \
	fi
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

# ============================================================================
# Build Target - Compile Configuration Parser, Operators, and Baselines
# ============================================================================
build: $(BUILD_DIR)/libcutekernellib_config.a $(BUILD_DIR)/libcutekernellib.a $(BUILD_DIR)/libcutekernellib_baselines.a

$(BUILD_DIR)/config_parser.o: $(SRC_DIR)/config/config_parser.cpp
	@echo "Compiling configuration parser..."
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -I$(YAML_CPP_DIR)/include -c $< -o $@
	@echo "  ✓ Configuration parser compiled"

$(BUILD_DIR)/libcutekernellib_config.a: $(BUILD_DIR)/config_parser.o
	@echo "Creating configuration library..."
	ar rcs $@ $^
	ranlib $@
	@echo "  ✓ Library created: $@"

# Compile operators
$(BUILD_DIR)/elementwise_add.o: $(SRC_DIR)/operators/elementwise_add.cu
	@echo "Compiling elementwise_add operator..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
	@echo "  ✓ elementwise_add operator compiled"

$(BUILD_DIR)/gemm.o: $(SRC_DIR)/operators/gemm.cu
	@echo "Compiling gemm operator..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
	@echo "  ✓ gemm operator compiled"

$(BUILD_DIR)/gemm_wgmma.o: $(SRC_DIR)/operators/gemm_wgmma.cu
	@echo "Compiling gemm_wgmma operator (Hopper WGMMA)..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
	@echo "  ✓ gemm_wgmma operator compiled"

# Create operator library
OPERATOR_OBJS := $(patsubst %,$(BUILD_DIR)/%.o,$(OPERATORS))
OPERATOR_OBJS += $(BUILD_DIR)/gemm_wgmma.o

$(BUILD_DIR)/libcutekernellib.a: $(OPERATOR_OBJS)
	@echo "Creating operator library..."
	ar rcs $@ $^
	ranlib $@
	@echo "  ✓ Operator library created: $@"

# Compile baselines
$(BUILD_DIR)/cuda_elementwise_add.o: $(BASELINES_DIR)/cuda/cuda_elementwise_add.cu
	@echo "Compiling CUDA baseline: cuda_elementwise_add..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(BASELINES_DIR)/cuda -c $< -o $@
	@echo "  ✓ CUDA baseline cuda_elementwise_add compiled"

$(BUILD_DIR)/cuda_gemm.o: $(BASELINES_DIR)/cuda/cuda_gemm.cu
	@echo "Compiling CUDA baseline: cuda_gemm..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(BASELINES_DIR)/cuda -c $< -o $@
	@echo "  ✓ CUDA baseline cuda_gemm compiled"

# Create baseline library
BASELINE_OBJS := $(patsubst %,$(BUILD_DIR)/%.o,$(BASELINES))

$(BUILD_DIR)/libcutekernellib_baselines.a: $(BASELINE_OBJS)
	@echo "Creating baseline library..."
	ar rcs $@ $^
	ranlib $@
	@echo "  ✓ Baseline library created: $@"

# ============================================================================
# Benchmark Targets
# ============================================================================

# Compile benchmark runner
$(BUILD_DIR)/benchmark_runner.o: $(BENCHMARKS_DIR)/benchmark_runner.cpp
	@echo "Compiling benchmark runner..."
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@
	@echo "  ✓ Benchmark runner compiled"

bench-elementwise-add: $(BUILD_DIR)/bench_elementwise_add
	@echo "=========================================="
	@echo "Running Element-wise Add Benchmark"
	@echo "=========================================="
	@$(BUILD_DIR)/bench_elementwise_add

$(BUILD_DIR)/bench_elementwise_add: $(BENCHMARKS_DIR)/bench_elementwise_add.cu $(BUILD_DIR)/benchmark_runner.o $(BUILD_DIR)/libcutekernellib.a $(BUILD_DIR)/libcutekernellib_baselines.a $(BUILD_DIR)/libcutekernellib_config.a
	@echo "Compiling element-wise add benchmark..."
	@$(MKDIR) $(BUILD_DIR)
	@$(MKDIR) $(BENCH_RESULTS_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		-I$(BENCHMARKS_DIR) \
		-I$(BASELINES_DIR)/cuda \
		$< $(BUILD_DIR)/benchmark_runner.o \
		-L$(BUILD_DIR) -lcutekernellib -lcutekernellib_baselines -lcutekernellib_config \
		-L$(YAML_CPP_DIR)/build -lyaml-cpp \
		-o $@
	@echo "  ✓ Benchmark program compiled"

bench-gemm: $(BUILD_DIR)/bench_gemm
	@echo "=========================================="
	@echo "Running GEMM Benchmark"
	@echo "=========================================="
	@$(BUILD_DIR)/bench_gemm

$(BUILD_DIR)/bench_gemm: $(BENCHMARKS_DIR)/bench_gemm.cu $(BUILD_DIR)/benchmark_runner.o $(BUILD_DIR)/libcutekernellib.a $(BUILD_DIR)/libcutekernellib_baselines.a $(BUILD_DIR)/libcutekernellib_config.a
	@echo "Compiling GEMM benchmark..."
	@$(MKDIR) $(BUILD_DIR)
	@$(MKDIR) $(BENCH_RESULTS_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		-I$(BENCHMARKS_DIR) \
		-I$(BASELINES_DIR)/cuda \
		$< $(BUILD_DIR)/benchmark_runner.o \
		-L$(BUILD_DIR) -lcutekernellib -lcutekernellib_baselines -lcutekernellib_config \
		-L$(YAML_CPP_DIR)/build -lyaml-cpp \
		-lcublas \
		-o $@
	@echo "  ✓ Benchmark program compiled"

# Benchmark all operators and generate summary report
bench-all:
	@echo "=========================================="
	@echo "Running All Benchmarks"
	@echo "=========================================="
	@$(MKDIR) $(BENCH_RESULTS_DIR)
	@echo ""
	@echo "Running benchmarks for all operators..."
	@for op in $(OPERATORS); do \
		echo ""; \
		echo "Benchmarking operator: $$op"; \
		target=$$(echo $$op | tr '_' '-'); \
		$(MAKE) bench-$$target || exit 1; \
	done
	@echo ""
	@echo "Generating summary report..."
	@if [ -f "scripts/generate_report.py" ]; then \
		python3 scripts/generate_report.py $(BENCH_RESULTS_DIR)/*_results.csv -o $(BENCH_RESULTS_DIR)/summary.md || \
		{ echo "  ⚠ Warning: Report generation failed, but benchmarks completed successfully"; }; \
		if [ -f "$(BENCH_RESULTS_DIR)/summary.md" ]; then \
			echo "  ✓ Summary report generated: $(BENCH_RESULTS_DIR)/summary.md"; \
		fi; \
	else \
		echo "  ⚠ Warning: Report generator not found at scripts/generate_report.py"; \
		echo "  Individual CSV results are available in $(BENCH_RESULTS_DIR)/"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "All Benchmarks Complete!"
	@echo "=========================================="
	@echo "Results available in:"
	@ls -lh $(BENCH_RESULTS_DIR)/*.csv 2>/dev/null || echo "  No CSV files found"
	@if [ -f "$(BENCH_RESULTS_DIR)/summary.md" ]; then \
		echo "  Summary: $(BENCH_RESULTS_DIR)/summary.md"; \
	fi
	@echo "=========================================="

# ============================================================================
# Test Targets
# ============================================================================

test-gemm-dispatch: $(BUILD_DIR)/test_gemm_dispatch
	@echo "=========================================="
	@echo "Running GEMM Dispatch Correctness Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_gemm_dispatch

$(BUILD_DIR)/test_gemm_dispatch: tests/test_gemm_dispatch.cu $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling GEMM dispatch test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-lcublas \
		-o $@
	@echo "  ✓ Test program compiled"

test-gemm-simple: $(BUILD_DIR)/test_gemm_simple
	@echo "=========================================="
	@echo "Running Simple GEMM Dispatch Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_gemm_simple

$(BUILD_DIR)/test_gemm_simple: tests/test_gemm_simple.cu $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling simple GEMM test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-o $@
	@echo "  ✓ Test program compiled"

test-init-methods: $(BUILD_DIR)/test_init_methods
	@echo "=========================================="
	@echo "Running Initialization Methods Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_init_methods

$(BUILD_DIR)/test_init_methods: tests/test_init_methods.cu $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling initialization methods test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-lcublas \
		-o $@
	@echo "  ✓ Test program compiled"

debug-cublas: $(BUILD_DIR)/debug_cublas
	@echo "=========================================="
	@echo "Running cuBLAS Configuration Debug"
	@echo "=========================================="
	@$(BUILD_DIR)/debug_cublas

$(BUILD_DIR)/debug_cublas: tests/debug_cublas.cu $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling cuBLAS debug test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-lcublas \
		-o $@
	@echo "  ✓ Test program compiled"

# ============================================================================
# Profiling Targets - Nsight Compute Integration
# ============================================================================

# Auto-detect Nsight Compute (ncu)
NCU := $(shell which ncu 2>/dev/null)
ifeq ($(NCU),)
    NCU := $(shell which nv-nsight-cu-cli 2>/dev/null)
endif
ifeq ($(NCU),)
    # Try to find ncu in CUDA installation
    ifneq ($(CUDA_PATH),)
        NCU_SEARCH_PATHS := $(CUDA_PATH)/bin/ncu \
                           $(CUDA_PATH)/nsight-compute/ncu \
                           $(CUDA_PATH)/../nsight-compute/ncu \
                           /usr/local/cuda/bin/ncu \
                           /opt/nvidia/nsight-compute/ncu
        NCU := $(firstword $(foreach path,$(NCU_SEARCH_PATHS),$(wildcard $(path))))
    endif
endif

# Auto-detect nvidia-smi for driver check
NVIDIA_SMI := $(shell which nvidia-smi 2>/dev/null)
ifeq ($(NVIDIA_SMI),)
    NVIDIA_SMI_SEARCH_PATHS := /usr/bin/nvidia-smi \
                               /usr/local/cuda/bin/nvidia-smi \
                               $(CUDA_PATH)/bin/nvidia-smi
    NVIDIA_SMI := $(firstword $(foreach path,$(NVIDIA_SMI_SEARCH_PATHS),$(wildcard $(path))))
endif

tune-elementwise-add: $(BUILD_DIR)/bench_elementwise_add
	@echo "=========================================="
	@echo "Profiling Element-wise Add with Nsight Compute"
	@echo "=========================================="
	@if [ -z "$(NCU)" ] || [ ! -x "$(NCU)" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found or not executable"; \
		echo ""; \
		echo "  Searched locations:"; \
		echo "    - System PATH"; \
		echo "    - $(CUDA_PATH)/bin/ncu"; \
		echo "    - $(CUDA_PATH)/nsight-compute/ncu"; \
		echo "    - /usr/local/cuda/bin/ncu"; \
		echo ""; \
		echo "  Nsight Compute is required for profiling."; \
		echo "  Installation options:"; \
		echo "    1. Install with CUDA toolkit (recommended)"; \
		echo "    2. Download standalone from:"; \
		echo "       https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		echo "  After installation, ensure ncu is in PATH:"; \
		echo "    export PATH=/usr/local/cuda/bin:\$$PATH"; \
		echo "  Or set NCU variable:"; \
		echo "    make tune-elementwise-add NCU=/path/to/ncu"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Using NCU: $(NCU)"
	@$(MKDIR) $(PROFILING_DIR)
	@echo ""
	@echo "Running ncu profiler (this may take several minutes)..."
	@echo "Output will be saved to: $(PROFILING_DIR)/elementwise_add_profile.ncu-rep"
	@echo ""
	@$(NCU) --set full \
		--export $(PROFILING_DIR)/elementwise_add_profile \
		--force-overwrite \
		--target-processes all \
		$(BUILD_DIR)/bench_elementwise_add || \
		{ echo ""; \
		  echo "  ⚠ Warning: Profiling completed with errors (may be due to driver compatibility)"; \
		  echo "  Profile data was still collected and saved."; \
		  echo ""; \
		}
	@echo ""
	@echo "Generating human-readable summary..."
	@if [ -f "$(PROFILING_DIR)/elementwise_add_profile.ncu-rep" ]; then \
		$(NCU) --import $(PROFILING_DIR)/elementwise_add_profile.ncu-rep \
			--page raw \
			--csv > $(PROFILING_DIR)/elementwise_add_summary.txt 2>&1 || \
		$(NCU) --import $(PROFILING_DIR)/elementwise_add_profile.ncu-rep \
			> $(PROFILING_DIR)/elementwise_add_summary.txt 2>&1 || \
		echo "Summary generation skipped (ncu-rep file may be incomplete)"; \
	else \
		echo "  ⚠ Warning: Profile report not found, skipping summary generation"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "Profiling Complete!"
	@echo "=========================================="
	@if [ -f "$(PROFILING_DIR)/elementwise_add_profile.ncu-rep" ]; then \
		echo "Results saved to:"; \
		echo "  - NCU Report:  $(PROFILING_DIR)/elementwise_add_profile.ncu-rep"; \
		if [ -f "$(PROFILING_DIR)/elementwise_add_summary.txt" ]; then \
			echo "  - Summary:     $(PROFILING_DIR)/elementwise_add_summary.txt"; \
		fi; \
		echo ""; \
		echo "To view detailed report in Nsight Compute GUI:"; \
		echo "  ncu-ui $(PROFILING_DIR)/elementwise_add_profile.ncu-rep"; \
	else \
		echo "  ✗ Profiling failed - no report generated"; \
	fi
	@echo "=========================================="

tune-gemm: $(BUILD_DIR)/bench_gemm
	@echo "=========================================="
	@echo "Profiling GEMM with Nsight Compute"
	@echo "=========================================="
	@if [ -z "$(NCU)" ] || [ ! -x "$(NCU)" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found or not executable"; \
		echo ""; \
		echo "  Searched locations:"; \
		echo "    - System PATH"; \
		echo "    - $(CUDA_PATH)/bin/ncu"; \
		echo "    - $(CUDA_PATH)/nsight-compute/ncu"; \
		echo "    - /usr/local/cuda/bin/ncu"; \
		echo ""; \
		echo "  Nsight Compute is required for profiling."; \
		echo "  Installation options:"; \
		echo "    1. Install with CUDA toolkit (recommended)"; \
		echo "    2. Download standalone from:"; \
		echo "       https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		echo "  After installation, ensure ncu is in PATH:"; \
		echo "    export PATH=/usr/local/cuda/bin:\$$PATH"; \
		echo "  Or set NCU variable:"; \
		echo "    make tune-gemm NCU=/path/to/ncu"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Using NCU: $(NCU)"
	@$(MKDIR) $(PROFILING_DIR)
	@echo ""
	@echo "Running ncu profiler (this may take several minutes)..."
	@echo "Output will be saved to: $(PROFILING_DIR)/gemm_profile.ncu-rep"
	@echo ""
	@$(NCU) --set full \
		--section LaunchStats \
		--section Occupancy \
		--section SpeedOfLight \
		--section MemoryWorkloadAnalysis \
		--section ComputeWorkloadAnalysis \
		--export $(PROFILING_DIR)/gemm_profile \
		--force-overwrite \
		--target-processes all \
		--kernel-name regex:.*gemm.* \
		$(BUILD_DIR)/bench_gemm || \
	{ echo ""; \
	  echo "  ⚠ Warning: Profiling completed with errors (may be due to driver compatibility)"; \
	  echo "  Profile data was still collected and saved."; \
	  echo ""; \
	}
	@echo ""
	@echo "Generating human-readable summary..."
	@if [ -f "$(PROFILING_DIR)/gemm_profile.ncu-rep" ]; then \
		echo "Generating detailed report with launch statistics and occupancy..." && \
		$(NCU) --import $(PROFILING_DIR)/gemm_profile.ncu-rep \
			> $(PROFILING_DIR)/gemm_summary.txt 2>&1 && \
		echo "Summary generation completed"; \
	else \
		echo "  ⚠ Warning: Profile report not found, skipping summary generation"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "Profiling Complete!"
	@echo "=========================================="
	@if [ -f "$(PROFILING_DIR)/gemm_profile.ncu-rep" ]; then \
		echo "Results saved to:"; \
		echo "  - NCU Report:     $(PROFILING_DIR)/gemm_profile.ncu-rep"; \
		if [ -f "$(PROFILING_DIR)/gemm_summary.txt" ]; then \
			echo "  - Summary:        $(PROFILING_DIR)/gemm_summary.txt"; \
			echo ""; \
			echo "Quick preview of launch statistics:"; \
			grep -E "(Block Size|Grid Size|Registers Per Thread|Theoretical Occupancy|Achieved Occupancy)" \
				$(PROFILING_DIR)/gemm_summary.txt | head -10 || true; \
		fi; \
		echo ""; \
		echo "To view full report:"; \
		echo "  cat $(PROFILING_DIR)/gemm_summary.txt"; \
		echo ""; \
		echo "To view in Nsight Compute GUI:"; \
		echo "  ncu-ui $(PROFILING_DIR)/gemm_profile.ncu-rep"; \
	else \
		echo "  ✗ Profiling failed - no report generated"; \
	fi
	@echo "=========================================="

tune-all:
	@echo "=========================================="
	@echo "Profiling All Operators"
	@echo "=========================================="
	@for op in $(OPERATORS); do \
		echo ""; \
		echo "Profiling operator: $$op"; \
		target=$$(echo $$op | tr '_' '-'); \
		$(MAKE) tune-$$target || exit 1; \
	done
	@echo ""
	@echo "=========================================="
	@echo "All Operators Profiled Successfully!"
	@echo "=========================================="
	@echo "Profile reports available in: $(PROFILING_DIR)/"
	@ls -lh $(PROFILING_DIR)/*.ncu-rep 2>/dev/null || true
	@echo "=========================================="

clean:
	@echo "Cleaning build artifacts..."
	$(RM) $(BUILD_DIR)
	$(RM) $(BENCH_RESULTS_DIR)/*.csv
	$(RM) $(PROFILING_DIR)/*.ncu-rep
	$(RM) $(PROFILING_DIR)/*.txt
	@echo "Clean complete."

