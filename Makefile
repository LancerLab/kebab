# CuTeKernelLib Makefile
# High-performance kernel library using NVIDIA CUTLASS CuTe

.PHONY: all setup build clean help test gpu-info gpu-reset

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
YAML_CPP_DIR := $(THIRD_PARTY_DIR)/yaml-cpp

# ============================================================================
# Operators (Define list for auto-generation of targets)
# ============================================================================
OPERATORS := elementwise_add gemm
# Convert underscores to hyphens for target names
OPERATORS_TGT := $(subst _,-,$(OPERATORS))

# ============================================================================
# Microbenchmarks (Define list for auto-generation of targets)
# ============================================================================
MICROBENCHS := copy_gmem_to_smem mma_wgmma mma_mma_sync mma_wmma_sync copy_gmem_to_smem_2d_tiling
# Convert underscores to hyphens for target names
MICROBENCHS_TGT := $(subst _,-,$(MICROBENCHS))

# ============================================================================
# Baselines
# ============================================================================
BASELINES := cuda_elementwise_add cuda_gemm

# Build mode
BUILD_MODE ?= release

# ============================================================================
# Conditional Environment Detection
# ============================================================================
# Only detect CUDA environment for targets that need it
# This avoids expensive detection for simple targets like clean, help, etc.

# List of targets that DON'T need CUDA detection
SIMPLE_TARGETS := clean help setup

# Check if we're running a simple target
NEED_CUDA := yes
ifneq ($(MAKECMDGOALS),)
    ifeq ($(filter-out $(SIMPLE_TARGETS),$(MAKECMDGOALS)),)
        NEED_CUDA := no
    endif
endif

# ============================================================================
# GPU Architecture Detection
# ============================================================================
ifeq ($(NEED_CUDA),yes)
ifndef CUDA_ARCH
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
    
    # Convert sm_90 to sm_90a for WGMMA support
    ifeq ($(CUDA_ARCH),sm_90)
        CUDA_ARCH := sm_90a
        $(info Converting sm_90 to sm_90a for WGMMA support)
    endif
endif
endif

# ============================================================================
# CUDA Toolkit Detection
# ============================================================================
ifeq ($(NEED_CUDA),yes)
ifndef CUDA_PATH
    # Step 1: Detect driver-supported CUDA version
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
    
    # Step 3: Find the best compatible CUDA version
    BEST_CUDA_PATH := 
    BEST_CUDA_VERSION := 0.0
    
    # Check if we detected driver version
    ifneq ($(DRIVER_CUDA_VERSION),)
        $(info Detected driver supports CUDA version: $(DRIVER_CUDA_VERSION))
        
        # Convert driver version to comparable number (e.g., 12.8 -> 1208)
        DRIVER_VERSION_NUM := $(shell echo $(DRIVER_CUDA_VERSION) | awk -F. '{printf "%d%02d", $$1, $$2}')
        
        # Check each available CUDA installation
        $(foreach path,$(AVAILABLE_CUDA_PATHS), \
            $(eval CUDA_VER := $(shell echo $(path) | sed -n 's|.*/cuda-\([0-9]\+\.[0-9]\+\).*|\1|p')) \
            $(if $(CUDA_VER), \
                $(eval CUDA_VER_NUM := $(shell echo $(CUDA_VER) | awk -F. '{printf "%d%02d", $$1, $$2}')) \
                $(if $(shell [ $(CUDA_VER_NUM) -le $(DRIVER_VERSION_NUM) ] && [ $(CUDA_VER_NUM) -gt $(shell echo $(BEST_CUDA_VERSION) | awk -F. '{printf "%d%02d", $$1, $$2}') ] && echo yes), \
                    $(eval BEST_CUDA_PATH := $(path)) \
                    $(eval BEST_CUDA_VERSION := $(CUDA_VER)) \
                ) \
            ) \
        )
        
        # Use the best compatible version
        ifneq ($(BEST_CUDA_PATH),)
            CUDA_PATH := $(BEST_CUDA_PATH)
            $(info Selected compatible CUDA toolkit: $(CUDA_PATH) (version $(BEST_CUDA_VERSION)))
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
endif
endif

# ============================================================================
# Compiler and Flags (Only set when CUDA is needed)
# ============================================================================
ifeq ($(NEED_CUDA),yes)
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

# Build mode specific flags
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
endif

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
	@echo ""
	@echo "Available targets:"
	@echo "  make setup              - Install dependencies (CuTe, yaml-cpp)"
	@echo "  make build              - Compile library and operators"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make help               - Show this help message"
	@echo "  make gpu-info           - Display GPU and CUDA information"
	@echo "  make gpu-reset          - Kill GPU processes and reset GPU state"
	@echo ""
	@echo "Test targets:"
	@echo "  make test               - Run all tests (uses bench-gemm with verification)"
	@echo ""
	@echo "Benchmark targets (after build):"
	@$(foreach op,$(OPERATORS_TGT),echo "  make bench-$(op)";)
	@echo "  make bench-all          - Run all benchmarks and generate summary"
	@echo ""
	@echo "Microbenchmark targets (after build):"
	@$(foreach mb,$(MICROBENCHS_TGT),echo "  make mbench-$(mb)";)
	@echo ""
	@echo "Dump targets (PTX/SASS with human-readable kernel names):"
	@$(foreach mb,$(MICROBENCHS_TGT),echo "  make mdump-$(mb)";)
	@$(foreach op,$(OPERATORS_TGT),echo "  make dump-$(op)-cute";)
	@$(foreach op,$(OPERATORS_TGT),echo "  make dump-$(op)-cuda";)
	@$(foreach op,$(OPERATORS_TGT),echo "  make dump-$(op)-ref";)
	@echo ""
	@echo "Profiling targets (Nsight Compute):"
	@echo "  Microbenchmarks (per-kernel reports):"
	@$(foreach mb,$(MICROBENCHS_TGT),echo "    make mtune-$(mb)";)
	@echo "    make mtune-all        - Profile all microbenchmarks"
	@echo "  Operators (single-run, no loops):"
	@$(foreach op,$(OPERATORS_TGT),echo "    make tune-$(op)-cute";)
	@$(foreach op,$(OPERATORS_TGT),echo "    make tune-$(op)-cuda";)
	@$(foreach op,$(OPERATORS_TGT),echo "    make tune-$(op)-ref";)
	@echo "    make tune-all         - Profile all operators (cute + cuda + ref)"
	@echo ""
	@echo "Debug/Run targets (for development):"
	@$(foreach op,$(OPERATORS_TGT),echo "  make run-$(op)-cute";)
	@$(foreach op,$(OPERATORS_TGT),echo "  make run-$(op)-cuda";)
	@$(foreach op,$(OPERATORS_TGT),echo "  make run-$(op)-ref";)
	@echo ""
	@echo "Report targets:"
	@echo "  make report             - Generate text reports from operator NCU profiles"
	@echo "  make mreport            - Generate text reports from microbench NCU profiles"
	@echo "=========================================="

# Display detected GPU info
gpu-info:
	@echo "=========================================="
	@echo "GPU Information"
	@echo "=========================================="
	@NVIDIA_SMI=$$(which nvidia-smi 2>/dev/null); \
	if [ -n "$$NVIDIA_SMI" ] && [ -x "$$NVIDIA_SMI" ]; then \
		$$NVIDIA_SMI --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader; \
	else \
		echo "  ✗ nvidia-smi not found or not executable"; \
		echo "  Please ensure NVIDIA drivers are installed"; \
	fi
	@echo ""
	@COMPUTE_CAP=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.'); \
	if [ -n "$$COMPUTE_CAP" ]; then \
		echo "Detected CUDA Architecture: sm_$$COMPUTE_CAP"; \
	else \
		echo "CUDA Architecture: Not detected"; \
	fi
	@CUDA_PATH_DETECT=$$(which nvcc 2>/dev/null | sed 's|/bin/nvcc||'); \
	if [ -z "$$CUDA_PATH_DETECT" ]; then \
		CUDA_PATH_DETECT=$$(ls -d /usr/local/cuda /usr/local/cuda-* /opt/cuda 2>/dev/null | head -n1); \
	fi; \
	if [ -n "$$CUDA_PATH_DETECT" ]; then \
		echo "CUDA Path: $$CUDA_PATH_DETECT"; \
		echo "NVCC: $$CUDA_PATH_DETECT/bin/nvcc"; \
	else \
		echo "CUDA Path: Not found"; \
	fi
	@NCU=$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
	if [ -n "$$NCU" ]; then \
		echo "NCU: $$NCU"; \
	else \
		echo "NCU: Not found"; \
	fi
	@echo "=========================================="

# Reset GPU - Kill GPU processes and reset GPU state
gpu-reset:
	@echo "=========================================="
	@echo "GPU Reset - Killing GPU Processes"
	@echo "=========================================="
	@echo ""
	@echo "[1/3] Checking for GPU processes using fuser..."
	@FUSER=$$(which fuser 2>/dev/null); \
	if [ -z "$$FUSER" ]; then \
		echo "  ✗ ERROR: fuser command not found"; \
		echo "  Please install psmisc package:"; \
		echo "    Ubuntu/Debian: sudo apt-get install psmisc"; \
		echo "    Fedora/RHEL:   sudo dnf install psmisc"; \
		echo "    Arch Linux:    sudo pacman -S psmisc"; \
		exit 1; \
	fi
	@echo "  ✓ fuser found"
	@echo ""
	@echo "[2/3] Listing GPU processes..."
	@FUSER=$$(which fuser 2>/dev/null); \
	GPU_PROCS=$$($$FUSER /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -u); \
	if [ -z "$$GPU_PROCS" ]; then \
		echo "  ℹ No GPU processes found"; \
	else \
		echo "  Found GPU processes:"; \
		for pid in $$GPU_PROCS; do \
			if [ -n "$$pid" ] && [ "$$pid" != "kernel" ]; then \
				CMD=$$(ps -p $$pid -o comm= 2>/dev/null || echo "unknown"); \
				echo "    PID: $$pid ($$CMD)"; \
			fi; \
		done; \
	fi
	@echo ""
	@echo "[3/3] Killing GPU processes..."
	@FUSER=$$(which fuser 2>/dev/null); \
	GPU_PROCS=$$($$FUSER /dev/nvidia* 2>/dev/null | tr ' ' '\n' | sort -u); \
	KILLED=0; \
	for pid in $$GPU_PROCS; do \
		if [ -n "$$pid" ] && [ "$$pid" != "kernel" ]; then \
			if kill -9 $$pid 2>/dev/null; then \
				echo "  ✓ Killed process $$pid"; \
				KILLED=$$((KILLED + 1)); \
			else \
				echo "  ⚠ Failed to kill process $$pid (may require sudo)"; \
			fi; \
		fi; \
	done; \
	if [ $$KILLED -eq 0 ]; then \
		echo "  ℹ No processes were killed"; \
	else \
		echo "  ✓ Killed $$KILLED process(es)"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "GPU Reset Complete!"
	@echo "=========================================="
	@echo ""
	@echo "Note: If you see 'Failed to kill process' messages,"
	@echo "you may need to run with sudo:"
	@echo "  sudo make gpu-reset"
	@echo "=========================================="

# ============================================================================
# Setup Target - Install Dependencies
# ============================================================================

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
	@echo "[4/4] Checking yaml-cpp installation..."
	@if pkg-config --exists yaml-cpp 2>/dev/null; then \
	echo "  ✓ yaml-cpp found via pkg-config"; \
	pkg-config --modversion yaml-cpp | xargs -I {} echo "    Version: {}"; \
	elif [ -f "/usr/include/yaml-cpp/yaml.h" ] || [ -f "/usr/local/include/yaml-cpp/yaml.h" ]; then \
	echo "  ✓ yaml-cpp headers found in system"; \
	else \
	echo "  ✗ ERROR: yaml-cpp not found in system"; \
	echo ""; \
	echo "  Please install yaml-cpp development package:"; \
	echo ""; \
	echo "  Ubuntu/Debian:"; \
	echo "    sudo apt-get install libyaml-cpp-dev"; \
	echo ""; \
	echo "  Fedora/RHEL:"; \
	echo "    sudo dnf install yaml-cpp-devel"; \
	echo ""; \
	echo "  Arch Linux:"; \
	echo "    sudo pacman -S yaml-cpp"; \
	echo ""; \
	echo "  macOS:"; \
	echo "    brew install yaml-cpp"; \
	echo ""; \
	exit 1; \
	fi
	@echo "  ✓ yaml-cpp check complete"
	@echo ""
	@echo "=========================================="
	@echo "Setup complete!"
	@echo "=========================================="
	@echo ""
	@echo "Dependencies:"
	@echo "  - CuTe headers:  $(CUTE_DIR)/include"
	@echo "  - yaml-cpp:      System installation"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make build' to compile the library"
	@echo "  2. Run 'make bench-<operator>' to benchmark operators"
	@echo "=========================================="

# ============================================================================
# Build Target - Use CMake as unified build system
# ============================================================================
build: cmake-build

cmake-configure:
	@echo "=========================================="
	@echo "Configuring CMake build system..."
	@echo "=========================================="
	@echo "Using CUDA_ARCH: $(CUDA_ARCH)"
	@echo "Using CUDA_PATH: $(CUDA_PATH)"
	@$(MKDIR) $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=$$(echo $(BUILD_MODE) | sed 's/debug/Debug/;s/release/Release/') \
		-DCUDA_ARCH=$(CUDA_ARCH) \
		-DCUDA_PATH=$(CUDA_PATH) \
		../kebab
	@echo "  ✓ CMake configuration complete"

cmake-build: cmake-configure
	@echo "=========================================="
	@echo "Building with CMake..."
	@echo "=========================================="
	@cd $(BUILD_DIR) && make -j$$(nproc)
	@echo "  ✓ Build complete"

# ============================================================================
# Benchmark Targets (Auto-generated from OPERATORS list)
# ============================================================================

# Define a template for benchmark targets
define BENCH_TEMPLATE
bench-$(1): build
	@echo "=========================================="
	@echo "Running $(subst -,_,$(1)) Benchmark"
	@echo "=========================================="
	@$$(BUILD_DIR)/lib/benchmark/bench_$(subst -,_,$(1))
	@echo "  ✓ Benchmark complete"
endef

# Generate benchmark targets for each operator
$(foreach op,$(OPERATORS_TGT),$(eval $(call BENCH_TEMPLATE,$(op))))

# Benchmark all operators and generate summary report
bench-all:
	@echo "=========================================="
	@echo "Running All Benchmarks"
	@echo "=========================================="
	@$(MKDIR) $(BENCH_RESULTS_DIR)
	@echo ""
	@echo "Running benchmarks for all operators..."
	@$(foreach op,$(OPERATORS_TGT), \
		echo ""; \
		echo "Benchmarking operator: $(op)"; \
		$(MAKE) bench-$(op) || exit 1; \
	)
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
# Microbenchmark Targets (Auto-generated from MICROBENCHS list)
# ============================================================================

# Directory for kernel dump (PTX/SASS)
DUMP_DIR := dump

# Define a template for microbenchmark targets (with hyphens)
define MBENCH_TEMPLATE
mbench-$(1): build
	@echo "=========================================="
	@echo "Running $(subst -,_,$(1)) Microbenchmark"
	@echo "=========================================="
	@$$(BUILD_DIR)/lib/microbench/mbench_$(subst -,_,$(1))
	@echo "  ✓ Microbenchmark complete"
endef

# Define a template for microbenchmark dump (PTX/SASS) targets with human-readable names
define MDUMP_TEMPLATE
mdump-$(1): build
	@echo "=========================================="
	@echo "Dumping $(subst -,_,$(1)) kernels (PTX/SASS)"
	@echo "=========================================="
	@$$(MKDIR) $$(DUMP_DIR)/microbench/$(subst -,_,$(1))
	@CUOBJDUMP=$$$$(which cuobjdump 2>/dev/null); \
	if [ -z "$$$$CUOBJDUMP" ]; then \
		echo "Searching for cuobjdump..."; \
		for base_path in /usr/local/cuda-* /usr/local/cuda $$(CUDA_PATH); do \
			if [ -d "$$$$base_path" ]; then \
				if [ -x "$$$$base_path/bin/cuobjdump" ]; then \
					CUOBJDUMP="$$$$base_path/bin/cuobjdump"; \
					break; \
				fi; \
			fi; \
		done; \
	fi; \
	if [ -z "$$$$CUOBJDUMP" ] || [ ! -x "$$$$CUOBJDUMP" ]; then \
		echo ""; \
		echo "  ✗ ERROR: cuobjdump not found"; \
		echo ""; \
		echo "  Please ensure CUDA toolkit is installed properly."; \
		echo ""; \
		exit 1; \
	fi; \
	BINARY=$$(BUILD_DIR)/lib/microbench/mbench_$(subst -,_,$(1)); \
	OUTDIR=$$(DUMP_DIR)/microbench/$(subst -,_,$(1)); \
	echo "Using cuobjdump: $$$$CUOBJDUMP"; \
	echo "Binary: $$$$BINARY"; \
	echo "Output: $$$$OUTDIR"; \
	echo ""; \
	echo "Extracting SASS (assembly)..."; \
	$$$$CUOBJDUMP --dump-sass $$$$BINARY > $$$$OUTDIR/all_kernels.sass; \
	echo "Extracting PTX (intermediate)..."; \
	$$$$CUOBJDUMP --dump-ptx $$$$BINARY > $$$$OUTDIR/all_kernels.ptx 2>/dev/null || echo "  (No embedded PTX found)"; \
	echo ""; \
	echo "Splitting SASS by kernel (with demangled names)..."; \
	( cd $$$$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.sass' all_kernels.sass '/Function :/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.sass; do \
		if [ -f "$$$$f" ]; then \
			mangled=$$$$(grep -oP 'Function : \K[^ ]+' "$$$$f" | head -1); \
			if [ -n "$$$$mangled" ]; then \
				demangled=$$$$(echo "$$$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$$$demangled" ] && [ "$$$$demangled" != "$$$$mangled" ]; then \
					mv "$$$$f" "$$$${demangled}.sass"; \
					echo "  $$$${demangled}.sass"; \
				else \
					mv "$$$$f" "$$$${mangled}.sass"; \
					echo "  $$$${mangled}.sass"; \
				fi; \
			else \
				rm -f "$$$$f"; \
			fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Splitting PTX by kernel (with demangled names)..."; \
	( cd $$$$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.ptx' all_kernels.ptx '/\.visible \.entry/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.ptx; do \
		if [ -f "$$$$f" ]; then \
			mangled=$$$$(grep -oP '\.visible \.entry \K[^(]+' "$$$$f" | head -1); \
			if [ -n "$$$$mangled" ]; then \
				demangled=$$$$(echo "$$$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$$$demangled" ] && [ "$$$$demangled" != "$$$$mangled" ]; then \
					if [ ! -f "$$$${demangled}.ptx" ]; then \
						echo "  $$$${demangled}.ptx"; \
					fi; \
					mv "$$$$f" "$$$${demangled}.ptx"; \
				else \
					if [ ! -f "$$$${mangled}.ptx" ]; then \
						echo "  $$$${mangled}.ptx"; \
					fi; \
					mv "$$$$f" "$$$${mangled}.ptx"; \
				fi; \
			else \
				rm -f "$$$$f"; \
			fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Generated files in $$$$OUTDIR/:"; \
	ls -lh $$$$OUTDIR/*.sass $$$$OUTDIR/*.ptx 2>/dev/null | grep -v all_kernels || true; \
	echo ""; \
	echo "  ✓ Dump complete"
endef

# Generate microbenchmark targets for each microbenchmark
$(foreach mb,$(MICROBENCHS_TGT),$(eval $(call MBENCH_TEMPLATE,$(mb))))

# Generate microbenchmark dump targets for each microbenchmark
$(foreach mb,$(MICROBENCHS_TGT),$(eval $(call MDUMP_TEMPLATE,$(mb))))

# Define a template for microbenchmark NCU profiling targets (per-kernel reports)
define MTUNE_TEMPLATE
mtune-$(1): build
	@echo "=========================================="
	@echo "NCU Profiling $(subst -,_,$(1)) Microbenchmark"
	@echo "=========================================="
	@NCU=$$$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
	if [ -z "$$$$NCU" ]; then \
		echo "Searching for Nsight Compute installation..."; \
		for base_path in /usr/local/cuda-* /opt/nvidia/nsight-compute /usr/local/cuda $$(CUDA_PATH); do \
			if [ -d "$$$$base_path" ]; then \
				for ncu_path in $$$$base_path/bin/ncu $$$$base_path/nsight-compute*/ncu $$$$base_path/nsight-compute/ncu; do \
					if [ -x "$$$$ncu_path" ]; then \
						NCU="$$$$ncu_path"; \
						break 2; \
					fi; \
				done; \
			fi; \
		done; \
	fi; \
	if [ -z "$$$$NCU" ] || [ ! -x "$$$$NCU" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found"; \
		echo ""; \
		echo "  Please install Nsight Compute:"; \
		echo "    https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		exit 1; \
	fi; \
	NCU_VERSION=$$$$($$$$NCU --version 2>/dev/null | grep -oP 'Version \K[0-9]+\.[0-9]+' | head -n1); \
	echo "Using NCU: $$$$NCU (Version $$$$NCU_VERSION)"; \
	$$(MKDIR) $$(PROFILING_DIR); \
	BINARY=$$(BUILD_DIR)/lib/microbench/mbench_$(subst -,_,$(1)); \
	echo ""; \
	echo "Binary: $$$$BINARY"; \
	echo ""; \
	if [ -n "$$(KERNEL)" ]; then \
		PROFILE_NAME="mbench_$(subst -,_,$(1))_$$(KERNEL)"; \
		echo "Profiling kernel: $$(KERNEL)"; \
		echo "Output: $$(PROFILING_DIR)/$$$$PROFILE_NAME.ncu-rep"; \
		echo ""; \
		$$$$NCU --set full \
			--export $$(PROFILING_DIR)/$$$$PROFILE_NAME \
			--force-overwrite \
			--kernel-name regex:"$$(KERNEL)" \
			--launch-count 1 \
			$$$$BINARY 2>&1 | grep -v "^==" || true; \
		echo ""; \
		if [ -f "$$(PROFILING_DIR)/$$$$PROFILE_NAME.ncu-rep" ]; then \
			echo "  ✓ Profile saved: $$(PROFILING_DIR)/$$$$PROFILE_NAME.ncu-rep"; \
		else \
			echo "  ✗ No profile generated"; \
		fi; \
	else \
		echo "Profiling each kernel separately..."; \
		echo "(Use 'make mtune-$(1) KERNEL=<name>' to profile specific kernel)"; \
		echo ""; \
		KERNELS=$$$$(nm $$$$BINARY 2>/dev/null | grep -oP 'kernel_[a-z_]+' | sort -u); \
		if [ -z "$$$$KERNELS" ]; then \
			echo "  Note: Could not detect kernel names, using defaults"; \
			KERNELS="kernel_copy_native kernel_copy_vectorized kernel_copy_ptx kernel_copy_cute"; \
		fi; \
		for kernel in $$$$KERNELS; do \
			PROFILE_NAME="mbench_$(subst -,_,$(1))_$$$$kernel"; \
			echo "Profiling: $$$$kernel"; \
			$$$$NCU --set full \
				--export $$(PROFILING_DIR)/$$$$PROFILE_NAME \
				--force-overwrite \
				--kernel-name regex:"$$$$kernel" \
				--launch-count 1 \
				$$$$BINARY 2>&1 | grep -v "^==" || true; \
			if [ -f "$$(PROFILING_DIR)/$$$$PROFILE_NAME.ncu-rep" ]; then \
				echo "  ✓ $$(PROFILING_DIR)/$$$$PROFILE_NAME.ncu-rep"; \
			else \
				echo "  ✗ Failed for $$$$kernel"; \
			fi; \
			echo ""; \
		done; \
	fi; \
	echo ""; \
	echo "Generated profiles:"; \
	ls -lh $$(PROFILING_DIR)/mbench_$(subst -,_,$(1))_*.ncu-rep 2>/dev/null || echo "  No profiles found"
endef

# Generate microbenchmark NCU profiling targets for each microbenchmark
$(foreach mb,$(MICROBENCHS_TGT),$(eval $(call MTUNE_TEMPLATE,$(mb))))

# mtune-all: Profile all microbenchmarks
mtune-all: build
	@echo "=========================================="
	@echo "Profiling All Microbenchmarks"
	@echo "=========================================="
	@$(foreach mb,$(MICROBENCHS_TGT), \
		echo ""; \
		echo "Profiling microbenchmark: $(mb)"; \
		$(MAKE) mtune-$(mb) || exit 1; \
	)
	@echo ""
	@echo "=========================================="
	@echo "All Microbenchmarks Profiled Successfully!"
	@echo "=========================================="
	@echo "Profile reports available in: $(PROFILING_DIR)/"
	@ls -lh $(PROFILING_DIR)/mbench_*.ncu-rep 2>/dev/null || true
	@echo "=========================================="

# mreport: Generate text reports from microbenchmark NCU profiles
mreport:
	@echo "=========================================="
	@echo "Generating Reports from Microbench NCU Profiles"
	@echo "=========================================="
	@$(MKDIR) reports
	@echo ""
	@echo "Processing microbenchmark NCU report files..."
	@NCU_FILES=$$(find $(PROFILING_DIR) -name "mbench_*.ncu-rep" -type f 2>/dev/null | sort); \
	if [ -z "$$NCU_FILES" ]; then \
		echo "  ⚠ No microbenchmark NCU report files found in $(PROFILING_DIR)/"; \
		echo "  Please run 'make mtune-all' first to generate profiles."; \
		exit 1; \
	fi; \
	COUNT=0; \
	SKIPPED=0; \
	for ncu_file in $$NCU_FILES; do \
		base_name=$$(basename "$$ncu_file" .ncu-rep); \
		report_file="reports/$${base_name}_report.txt"; \
		if [ -f "$$report_file" ]; then \
			echo "  Skipping: $$base_name (report already exists)"; \
			SKIPPED=$$((SKIPPED + 1)); \
		else \
			echo "  Exporting: $$base_name"; \
			NCU_BIN=$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
			if [ -z "$$NCU_BIN" ]; then \
				for base_path in /usr/local/cuda-* /opt/nvidia/nsight-compute /usr/local/cuda; do \
					if [ -d "$$base_path" ]; then \
						for ncu_path in $$base_path/bin/ncu $$base_path/nsight-compute*/ncu $$base_path/nsight-compute/ncu; do \
							if [ -x "$$ncu_path" ]; then \
								NCU_BIN="$$ncu_path"; \
								break 2; \
							fi; \
						done; \
					fi; \
				done; \
			fi; \
			if [ -z "$$NCU_BIN" ] || [ ! -x "$$NCU_BIN" ]; then \
				echo "    ✗ NCU not found, skipping"; \
				continue; \
			fi; \
			$$NCU_BIN --import "$$ncu_file" --page details > "$$report_file" 2>&1; \
			if [ -s "$$report_file" ]; then \
				echo "    ✓ $$report_file"; \
				COUNT=$$((COUNT + 1)); \
			else \
				echo "    ✗ Failed to generate report"; \
				rm -f "$$report_file"; \
			fi; \
		fi; \
	done; \
	echo ""; \
	echo "=========================================="
	@echo "Microbench Report Generation Complete!"
	@echo "=========================================="
	@echo ""
	@echo "All microbench reports:"
	@ls -lh reports/mbench_*.txt 2>/dev/null || echo "  No reports found"
	@echo "=========================================="

# ============================================================================
# Profiling Targets (Auto-generated from OPERATORS list)
# ============================================================================

# Generate profiling targets for CuTe implementation (single-run, no loops)
$(addprefix tune-,$(addsuffix -cute,$(OPERATORS_TGT))): tune-%-cute: build
	@echo "=========================================="
	@echo "Profiling $(subst -,_,$*) CuTe Kernel (Single Run)"
	@echo "=========================================="
	@NCU=$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
	if [ -z "$$NCU" ]; then \
		echo "Searching for Nsight Compute installation..."; \
		for base_path in /usr/local/cuda-* /opt/nvidia/nsight-compute /usr/local/cuda $(CUDA_PATH); do \
			if [ -d "$$base_path" ]; then \
				for ncu_path in $$base_path/bin/ncu $$base_path/nsight-compute*/ncu $$base_path/nsight-compute/ncu; do \
					if [ -x "$$ncu_path" ]; then \
						NCU="$$ncu_path"; \
						break 2; \
					fi; \
				done; \
			fi; \
		done; \
	fi; \
	if [ -z "$$NCU" ] || [ ! -x "$$NCU" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found"; \
		echo ""; \
		echo "  Please install Nsight Compute:"; \
		echo "    https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		exit 1; \
	fi; \
	NCU_VERSION=$$($$NCU --version 2>/dev/null | grep -oP 'Version \K[0-9]+\.[0-9]+' | head -n1); \
	echo "Using NCU: $$NCU (Version $$NCU_VERSION)"; \
	$(MKDIR) $(PROFILING_DIR); \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	PROFILE_NAME="$(subst -,_,$*)_cute_$${TIMESTAMP}_profile"; \
	echo ""; \
	echo "Running ncu profiler on $(subst -,_,$*) CuTe kernel (single run)..."; \
	echo "Output will be saved to: $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
	echo ""; \
	$$NCU --set full \
		--export $(PROFILING_DIR)/$${PROFILE_NAME} \
		--force-overwrite \
		--target-processes all \
		$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_cute || \
		{ echo ""; \
		  echo "  ⚠ Warning: Profiling completed with errors"; \
		  echo "  Profile data may still be available."; \
		  echo ""; \
		}; \
	echo ""; \
	echo "=========================================="; \
	echo "Profiling Complete!"; \
	echo "=========================================="; \
	if [ -f "$(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep" ]; then \
		echo "Results saved to:"; \
		echo "  - NCU Report:  $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
		echo ""; \
		echo "To view in Nsight Compute GUI:"; \
		echo "  ncu-ui $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
	else \
		echo "  ✗ Profiling failed - no report generated"; \
	fi; \
	echo "=========================================="

# Generate profiling targets for CUDA baseline implementation (single-run, no loops)
$(addprefix tune-,$(addsuffix -cuda,$(OPERATORS_TGT))): tune-%-cuda: build
	@echo "=========================================="
	@echo "Profiling $(subst -,_,$*) CUDA Baseline Kernel (Single Run)"
	@echo "=========================================="
	@NCU=$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
	if [ -z "$$NCU" ]; then \
		echo "Searching for Nsight Compute installation..."; \
		for base_path in /usr/local/cuda-* /opt/nvidia/nsight-compute /usr/local/cuda $(CUDA_PATH); do \
			if [ -d "$$base_path" ]; then \
				for ncu_path in $$base_path/bin/ncu $$base_path/nsight-compute*/ncu $$base_path/nsight-compute/ncu; do \
					if [ -x "$$ncu_path" ]; then \
						NCU="$$ncu_path"; \
						break 2; \
					fi; \
				done; \
			fi; \
		done; \
	fi; \
	if [ -z "$$NCU" ] || [ ! -x "$$NCU" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found"; \
		echo ""; \
		echo "  Please install Nsight Compute:"; \
		echo "    https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		exit 1; \
	fi; \
	NCU_VERSION=$$($$NCU --version 2>/dev/null | grep -oP 'Version \K[0-9]+\.[0-9]+' | head -n1); \
	echo "Using NCU: $$NCU (Version $$NCU_VERSION)"; \
	$(MKDIR) $(PROFILING_DIR); \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	PROFILE_NAME="$(subst -,_,$*)_cuda_$${TIMESTAMP}_profile"; \
	echo ""; \
	echo "Running ncu profiler on $(subst -,_,$*) CUDA baseline kernel (single run)..."; \
	echo "Output will be saved to: $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
	echo ""; \
	$$NCU --set full \
		--export $(PROFILING_DIR)/$${PROFILE_NAME} \
		--force-overwrite \
		--target-processes all \
		$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_cuda || \
		{ echo ""; \
		  echo "  ⚠ Warning: Profiling completed with errors"; \
		  echo "  Profile data may still be available."; \
		  echo ""; \
		}; \
	echo ""; \
	echo "=========================================="; \
	echo "Profiling Complete!"; \
	echo "=========================================="; \
	if [ -f "$(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep" ]; then \
		echo "Results saved to:"; \
		echo "  - NCU Report:  $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
		echo ""; \
		echo "To view in Nsight Compute GUI:"; \
		echo "  ncu-ui $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
	else \
		echo "  ✗ Profiling failed - no report generated"; \
	fi; \
	echo "=========================================="

# Generate reference profiling targets for each operator (single-run, no loops)
$(addsuffix -ref,$(addprefix tune-,$(OPERATORS_TGT))): tune-%-ref: build
	@echo "=========================================="
	@echo "Profiling $(subst -,_,$*) Reference Kernel (Single Run)"
	@echo "=========================================="
	@NCU=$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
	if [ -z "$$NCU" ]; then \
		echo "Searching for Nsight Compute installation..."; \
		for base_path in /usr/local/cuda-* /opt/nvidia/nsight-compute /usr/local/cuda $(CUDA_PATH); do \
			if [ -d "$$base_path" ]; then \
				for ncu_path in $$base_path/bin/ncu $$base_path/nsight-compute*/ncu $$base_path/nsight-compute/ncu; do \
					if [ -x "$$ncu_path" ]; then \
						NCU="$$ncu_path"; \
						break 2; \
					fi; \
				done; \
			fi; \
		done; \
	fi; \
	if [ -z "$$NCU" ] || [ ! -x "$$NCU" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found"; \
		echo ""; \
		echo "  Please install Nsight Compute:"; \
		echo "    https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		exit 1; \
	fi; \
	NCU_VERSION=$$($$NCU --version 2>/dev/null | grep -oP 'Version \K[0-9]+\.[0-9]+' | head -n1); \
	echo "Using NCU: $$NCU (Version $$NCU_VERSION)"; \
	$(MKDIR) $(PROFILING_DIR); \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	PROFILE_NAME="$(subst -,_,$*)_ref_$${TIMESTAMP}_profile"; \
	echo ""; \
	echo "Running ncu profiler on $(subst -,_,$*) reference kernel (single run)..."; \
	echo "Output will be saved to: $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
	echo ""; \
	$$NCU --set full \
		--export $(PROFILING_DIR)/$${PROFILE_NAME} \
		--force-overwrite \
		--target-processes all \
		$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_ref || \
		{ echo ""; \
		  echo "  ⚠ Warning: Profiling completed with errors"; \
		  echo "  Profile data may still be available."; \
		  echo ""; \
		}; \
	echo ""; \
	echo "=========================================="; \
	echo "Profiling Complete!"; \
	echo "=========================================="; \
	if [ -f "$(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep" ]; then \
		echo "Results saved to:"; \
		echo "  - NCU Report:  $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
		echo ""; \
		echo "To view in Nsight Compute GUI:"; \
		echo "  ncu-ui $(PROFILING_DIR)/$${PROFILE_NAME}.ncu-rep"; \
	else \
		echo "  ✗ Profiling failed - no report generated"; \
	fi; \
	echo "=========================================="

tune-all:
	@echo "=========================================="
	@echo "Profiling All Operators (CuTe + CUDA + Ref)"
	@echo "=========================================="
	@$(foreach op,$(OPERATORS_TGT), \
		echo ""; \
		echo "Profiling operator: $(op)"; \
		$(MAKE) tune-$(op)-cute || exit 1; \
		$(MAKE) tune-$(op)-cuda || exit 1; \
		$(MAKE) tune-$(op)-ref || exit 1; \
	)
	@echo ""
	@echo "=========================================="
	@echo "All Operators Profiled Successfully!"
	@echo "=========================================="
	@echo "Profile reports available in: $(PROFILING_DIR)/"
	@ls -lh $(PROFILING_DIR)/*.ncu-rep 2>/dev/null || true
	@echo "=========================================="

# ============================================================================
# Debug/Run Targets (Auto-generated from OPERATORS list)
# ============================================================================

# Generate run targets for CuTe implementation
$(addprefix run-,$(addsuffix -cute,$(OPERATORS_TGT))): run-%-cute: build
	@echo "=========================================="
	@echo "Running $(subst -,_,$*) CuTe Kernel (Single Run)"
	@echo "=========================================="
	@$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_cute
	@echo "=========================================="

# Generate run targets for CUDA baseline implementation
$(addprefix run-,$(addsuffix -cuda,$(OPERATORS_TGT))): run-%-cuda: build
	@echo "=========================================="
	@echo "Running $(subst -,_,$*) CUDA Kernel (Single Run)"
	@echo "=========================================="
	@$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_cuda
	@echo "=========================================="

# Generate run-ref targets for reference implementations
$(addprefix run-,$(addsuffix -ref,$(OPERATORS_TGT))): run-%-ref: build
	@echo "=========================================="
	@echo "Running $(subst -,_,$*) Reference Kernel (Single Run)"
	@echo "=========================================="
	@$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_ref
	@echo "=========================================="

# ============================================================================
# Operator Dump Targets (PTX/SASS with human-readable names)
# ============================================================================

# Generate dump targets for CuTe implementation
$(addprefix dump-,$(addsuffix -cute,$(OPERATORS_TGT))): dump-%-cute: build
	@echo "=========================================="
	@echo "Dumping $(subst -,_,$*) CuTe kernels (PTX/SASS)"
	@echo "=========================================="
	@$(MKDIR) $(DUMP_DIR)/operator/$(subst -,_,$*)_cute
	@CUOBJDUMP=$$(which cuobjdump 2>/dev/null); \
	if [ -z "$$CUOBJDUMP" ]; then \
		echo "Searching for cuobjdump..."; \
		for base_path in /usr/local/cuda-* /usr/local/cuda $(CUDA_PATH); do \
			if [ -d "$$base_path" ]; then \
				if [ -x "$$base_path/bin/cuobjdump" ]; then \
					CUOBJDUMP="$$base_path/bin/cuobjdump"; \
					break; \
				fi; \
			fi; \
		done; \
	fi; \
	if [ -z "$$CUOBJDUMP" ] || [ ! -x "$$CUOBJDUMP" ]; then \
		echo ""; \
		echo "  ✗ ERROR: cuobjdump not found"; \
		exit 1; \
	fi; \
	BINARY=$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_cute; \
	OUTDIR=$(DUMP_DIR)/operator/$(subst -,_,$*)_cute; \
	echo "Using cuobjdump: $$CUOBJDUMP"; \
	echo "Binary: $$BINARY"; \
	echo "Output: $$OUTDIR"; \
	echo ""; \
	echo "Extracting SASS (assembly)..."; \
	$$CUOBJDUMP --dump-sass $$BINARY > $$OUTDIR/all_kernels.sass; \
	echo "Extracting PTX (intermediate)..."; \
	$$CUOBJDUMP --dump-ptx $$BINARY > $$OUTDIR/all_kernels.ptx 2>/dev/null || echo "  (No embedded PTX found)"; \
	echo ""; \
	echo "Splitting SASS by kernel (with demangled names)..."; \
	( cd $$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.sass' all_kernels.sass '/Function :/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.sass; do \
		if [ -f "$$f" ]; then \
			mangled=$$(grep -oP 'Function : \K[^ ]+' "$$f" | head -1); \
			if [ -n "$$mangled" ]; then \
				demangled=$$(echo "$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$demangled" ] && [ "$$demangled" != "$$mangled" ]; then \
					mv "$$f" "$${demangled}.sass"; \
					echo "  $${demangled}.sass"; \
				else \
					mv "$$f" "$${mangled}.sass"; \
					echo "  $${mangled}.sass"; \
				fi; \
			else \
				rm -f "$$f"; \
			fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Splitting PTX by kernel (with demangled names)..."; \
	( cd $$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.ptx' all_kernels.ptx '/\.visible \.entry/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.ptx; do \
		if [ -f "$$f" ]; then \
			mangled=$$(grep -oP '\.visible \.entry \K[^(]+' "$$f" | head -1); \
			if [ -n "$$mangled" ]; then \
				demangled=$$(echo "$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$demangled" ] && [ "$$demangled" != "$$mangled" ]; then \
					if [ ! -f "$${demangled}.ptx" ]; then \
						echo "  $${demangled}.ptx"; \
					fi; \
					mv "$$f" "$${demangled}.ptx"; \
				else \
					if [ ! -f "$${mangled}.ptx" ]; then \
						echo "  $${mangled}.ptx"; \
					fi; \
					mv "$$f" "$${mangled}.ptx"; \
				fi; \
			else \
				rm -f "$$f"; \
			fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Generated files in $$OUTDIR/:"; \
	ls -lh $$OUTDIR/*.sass $$OUTDIR/*.ptx 2>/dev/null | grep -v all_kernels || true; \
	echo ""; \
	echo "  ✓ Dump complete"

# Generate dump targets for CUDA baseline implementation
$(addprefix dump-,$(addsuffix -cuda,$(OPERATORS_TGT))): dump-%-cuda: build
	@echo "=========================================="
	@echo "Dumping $(subst -,_,$*) CUDA baseline kernels (PTX/SASS)"
	@echo "=========================================="
	@$(MKDIR) $(DUMP_DIR)/operator/$(subst -,_,$*)_cuda
	@CUOBJDUMP=$$(which cuobjdump 2>/dev/null); \
	if [ -z "$$CUOBJDUMP" ]; then \
		for base_path in /usr/local/cuda-* /usr/local/cuda $(CUDA_PATH); do \
			if [ -d "$$base_path" ] && [ -x "$$base_path/bin/cuobjdump" ]; then \
				CUOBJDUMP="$$base_path/bin/cuobjdump"; \
				break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$CUOBJDUMP" ] || [ ! -x "$$CUOBJDUMP" ]; then \
		echo "  ✗ ERROR: cuobjdump not found"; exit 1; \
	fi; \
	BINARY=$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_cuda; \
	OUTDIR=$(DUMP_DIR)/operator/$(subst -,_,$*)_cuda; \
	echo "Using cuobjdump: $$CUOBJDUMP"; \
	echo "Binary: $$BINARY"; \
	echo "Output: $$OUTDIR"; \
	echo ""; \
	echo "Extracting SASS (assembly)..."; \
	$$CUOBJDUMP --dump-sass $$BINARY > $$OUTDIR/all_kernels.sass; \
	echo "Extracting PTX (intermediate)..."; \
	$$CUOBJDUMP --dump-ptx $$BINARY > $$OUTDIR/all_kernels.ptx 2>/dev/null || echo "  (No embedded PTX found)"; \
	echo ""; \
	echo "Splitting SASS by kernel (with demangled names)..."; \
	( cd $$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.sass' all_kernels.sass '/Function :/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.sass; do \
		if [ -f "$$f" ]; then \
			mangled=$$(grep -oP 'Function : \K[^ ]+' "$$f" | head -1); \
			if [ -n "$$mangled" ]; then \
				demangled=$$(echo "$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$demangled" ] && [ "$$demangled" != "$$mangled" ]; then \
					mv "$$f" "$${demangled}.sass"; echo "  $${demangled}.sass"; \
				else \
					mv "$$f" "$${mangled}.sass"; echo "  $${mangled}.sass"; \
				fi; \
			else rm -f "$$f"; fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Splitting PTX by kernel (with demangled names)..."; \
	( cd $$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.ptx' all_kernels.ptx '/\.visible \.entry/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.ptx; do \
		if [ -f "$$f" ]; then \
			mangled=$$(grep -oP '\.visible \.entry \K[^(]+' "$$f" | head -1); \
			if [ -n "$$mangled" ]; then \
				demangled=$$(echo "$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$demangled" ] && [ "$$demangled" != "$$mangled" ]; then \
					if [ ! -f "$${demangled}.ptx" ]; then echo "  $${demangled}.ptx"; fi; \
					mv "$$f" "$${demangled}.ptx"; \
				else \
					if [ ! -f "$${mangled}.ptx" ]; then echo "  $${mangled}.ptx"; fi; \
					mv "$$f" "$${mangled}.ptx"; \
				fi; \
			else rm -f "$$f"; fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Generated files in $$OUTDIR/:"; \
	ls -lh $$OUTDIR/*.sass $$OUTDIR/*.ptx 2>/dev/null | grep -v all_kernels || true; \
	echo ""; \
	echo "  ✓ Dump complete"

# Generate dump-ref targets for reference implementations
$(addprefix dump-,$(addsuffix -ref,$(OPERATORS_TGT))): dump-%-ref: build
	@echo "=========================================="
	@echo "Dumping $(subst -,_,$*) reference kernels (PTX/SASS)"
	@echo "=========================================="
	@$(MKDIR) $(DUMP_DIR)/operator/$(subst -,_,$*)_ref
	@CUOBJDUMP=$$(which cuobjdump 2>/dev/null); \
	if [ -z "$$CUOBJDUMP" ]; then \
		for base_path in /usr/local/cuda-* /usr/local/cuda $(CUDA_PATH); do \
			if [ -d "$$base_path" ] && [ -x "$$base_path/bin/cuobjdump" ]; then \
				CUOBJDUMP="$$base_path/bin/cuobjdump"; \
				break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$CUOBJDUMP" ] || [ ! -x "$$CUOBJDUMP" ]; then \
		echo "  ✗ ERROR: cuobjdump not found"; \
		exit 1; \
	fi; \
	BINARY=$(BUILD_DIR)/lib/benchmark/runonce_$(subst -,_,$*)_ref; \
	OUTDIR=$(DUMP_DIR)/operator/$(subst -,_,$*)_ref; \
	echo "Using cuobjdump: $$CUOBJDUMP"; \
	echo "Binary: $$BINARY"; \
	echo "Output: $$OUTDIR"; \
	echo ""; \
	echo "Extracting SASS (assembly)..."; \
	$$CUOBJDUMP --dump-sass $$BINARY > $$OUTDIR/all_kernels.sass; \
	echo "Extracting PTX (intermediate)..."; \
	$$CUOBJDUMP --dump-ptx $$BINARY > $$OUTDIR/all_kernels.ptx 2>/dev/null || echo "  (No embedded PTX found)"; \
	echo ""; \
	echo "Splitting SASS by kernel (with demangled names)..."; \
	( cd $$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.sass' all_kernels.sass '/Function :/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.sass; do \
		if [ -f "$$f" ]; then \
			mangled=$$(grep -oP 'Function : \K[^ ]+' "$$f" | head -1); \
			if [ -n "$$mangled" ]; then \
				demangled=$$(echo "$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$demangled" ] && [ "$$demangled" != "$$mangled" ]; then \
					mv "$$f" "$${demangled}.sass"; \
					echo "  $${demangled}.sass"; \
				else \
					mv "$$f" "$${mangled}.sass"; \
					echo "  $${mangled}.sass"; \
				fi; \
			else \
				rm -f "$$f"; \
			fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Splitting PTX by kernel (with demangled names)..."; \
	( cd $$OUTDIR && \
	csplit -z -f kernel_ -b '%02d.ptx' all_kernels.ptx '/\.visible \.entry/' '{*}' >/dev/null 2>&1 && \
	for f in kernel_*.ptx; do \
		if [ -f "$$f" ]; then \
			mangled=$$(grep -oP '\.visible \.entry \K[^(]+' "$$f" | head -1); \
			if [ -n "$$mangled" ]; then \
				demangled=$$(echo "$$mangled" | c++filt 2>/dev/null | sed 's/(.*//'); \
				if [ -n "$$demangled" ] && [ "$$demangled" != "$$mangled" ]; then \
					if [ ! -f "$${demangled}.ptx" ]; then \
						echo "  $${demangled}.ptx"; \
					fi; \
					mv "$$f" "$${demangled}.ptx"; \
				else \
					if [ ! -f "$${mangled}.ptx" ]; then \
						echo "  $${mangled}.ptx"; \
					fi; \
					mv "$$f" "$${mangled}.ptx"; \
				fi; \
			else \
				rm -f "$$f"; \
			fi; \
		fi; \
	done ); \
	echo ""; \
	echo "Generated files in $$OUTDIR/:"; \
	ls -lh $$OUTDIR/*.sass $$OUTDIR/*.ptx 2>/dev/null | grep -v all_kernels || true; \
	echo ""; \
	echo "  ✓ Dump complete"

# ============================================================================
# Report Generation Target - Extract Summary and Details from NCU reports
# ============================================================================
report:
	@echo "=========================================="
	@echo "Generating Reports from NCU Profiles"
	@echo "=========================================="
	@$(MKDIR) reports
	@echo ""
	@echo "Processing NCU report files..."
	@NCU_FILES=$$(find $(PROFILING_DIR) -name "*.ncu-rep" -type f 2>/dev/null | sort); \
	if [ -z "$$NCU_FILES" ]; then \
		echo "  ⚠ No NCU report files found in $(PROFILING_DIR)/"; \
		echo "  Please run 'make tune-all' first to generate profiles."; \
		exit 1; \
	fi; \
	COUNT=0; \
	SKIPPED=0; \
	for ncu_file in $$NCU_FILES; do \
		base_name=$$(basename "$$ncu_file" .ncu-rep); \
		report_file="reports/$${base_name}_report.txt"; \
		if [ -f "$$report_file" ]; then \
			echo "  Skipping: $$base_name (report already exists)"; \
			SKIPPED=$$((SKIPPED + 1)); \
		else \
			echo "  Exporting: $$base_name"; \
			NCU_BIN=$$(which ncu 2>/dev/null || which nv-nsight-cu-cli 2>/dev/null); \
			if [ -z "$$NCU_BIN" ]; then \
				for base_path in /usr/local/cuda-* /opt/nvidia/nsight-compute /usr/local/cuda; do \
					if [ -d "$$base_path" ]; then \
						for ncu_path in $$base_path/bin/ncu $$base_path/nsight-compute*/ncu $$base_path/nsight-compute/ncu; do \
							if [ -x "$$ncu_path" ]; then \
								NCU_BIN="$$ncu_path"; \
								break 2; \
							fi; \
						done; \
					fi; \
				done; \
			fi; \
			if [ -z "$$NCU_BIN" ] || [ ! -x "$$NCU_BIN" ]; then \
				echo "    ✗ NCU not found, skipping"; \
				continue; \
			fi; \
			$$NCU_BIN --import "$$ncu_file" --page details > "$$report_file" 2>&1; \
			if [ -s "$$report_file" ]; then \
				echo "    ✓ $$report_file"; \
				COUNT=$$((COUNT + 1)); \
			else \
				echo "    ✗ Failed to generate report"; \
				rm -f "$$report_file"; \
			fi; \
		fi; \
	done; \
	echo ""; \
	echo "=========================================="
	@echo "Report Generation Complete!"
	@echo "=========================================="
	@echo "Generated: $$COUNT new report(s)"
	@echo "Skipped: $$SKIPPED existing report(s)"
	@echo ""
	@echo "All reports:"
	@ls -lh reports/*.txt 2>/dev/null || echo "  No reports found"
	@echo "=========================================="

# ============================================================================
# Report Extraction Targets (Auto-generated from OPERATORS list)
# ============================================================================

# Generate report extraction targets for CuTe implementation
$(addprefix report-,$(addsuffix -cute,$(OPERATORS_TGT))): report-%-cute:
	@echo "=========================================="
	@echo "Extracting Report for $(subst -,_,$*) CuTe Kernel"
	@echo "=========================================="
	@$(MKDIR) reports
	@LATEST_REP=$$(ls -t $(PROFILING_DIR)/$(subst -,_,$*)_cute_*_profile.ncu-rep 2>/dev/null | head -n1); \
	if [ -z "$$LATEST_REP" ]; then \
		echo "Error: No profiling report found for $(subst -,_,$*) CuTe"; \
		echo "Please run 'make tune-$(subst _,-,$*)-cute' first"; \
		exit 1; \
	fi; \
	echo "Processing: $$LATEST_REP"; \
	python3 scripts/extract_ncu_report.py "$$LATEST_REP" reports/ || exit 1; \
	echo "=========================================="

# Generate report extraction targets for CUDA baseline implementation
$(addprefix report-,$(addsuffix -cuda,$(OPERATORS_TGT))): report-%-cuda:
	@echo "=========================================="
	@echo "Extracting Report for $(subst -,_,$*) CUDA Kernel"
	@echo "=========================================="
	@$(MKDIR) reports
	@LATEST_REP=$$(ls -t $(PROFILING_DIR)/$(subst -,_,$*)_cuda_*_profile.ncu-rep 2>/dev/null | head -n1); \
	if [ -z "$$LATEST_REP" ]; then \
		echo "Error: No profiling report found for $(subst -,_,$*) CUDA"; \
		echo "Please run 'make tune-$(subst _,-,$*)-cuda' first"; \
		exit 1; \
	fi; \
	echo "Processing: $$LATEST_REP"; \
	python3 scripts/extract_ncu_report.py "$$LATEST_REP" reports/ || exit 1; \
	echo "=========================================="

# Generate report extraction targets for reference kernels
$(addprefix report-,$(addsuffix -ref,$(OPERATORS_TGT))): report-%-ref:
	@echo "=========================================="
	@echo "Extracting Report for $(subst -,_,$*) Reference Kernel"
	@echo "=========================================="
	@$(MKDIR) reports
	@LATEST_REP=$$(ls -t $(PROFILING_DIR)/$(subst -,_,$*)_ref_*_profile.ncu-rep 2>/dev/null | head -n1); \
	if [ -z "$$LATEST_REP" ]; then \
		echo "Error: No reference profiling report found for $(subst -,_,$*)"; \
		echo "Please run 'make tune-$(subst _,-,$*)-ref' first"; \
		exit 1; \
	fi; \
	echo "Processing: $$LATEST_REP"; \
	python3 scripts/extract_ncu_report.py "$$LATEST_REP" reports/ || exit 1; \
	echo "=========================================="

# ============================================================================
# Test Target
# ============================================================================
test: bench-gemm

# ============================================================================
# Clean Target
# ============================================================================
clean:
	@echo "Cleaning build artifacts..."
	@$(RM) $(BUILD_DIR)
	@$(RM) $(BENCH_RESULTS_DIR)/*.csv
	@$(RM) $(PROFILING_DIR)/*.ncu-rep
	@$(RM) $(PROFILING_DIR)/*.txt
	@echo "Clean complete."
