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
# Find NVCC in standard CUDA installation paths (prefer CUDA 12 for compatibility)
NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda-12/bin/nvcc)
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
CUDA_INCLUDE := $(shell dirname $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc))/../include
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

# Create operator library
OPERATOR_OBJS := $(patsubst %,$(BUILD_DIR)/%.o,$(OPERATORS))

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
# Test Target - Test Configuration Parser, Operators, and Baselines
# ============================================================================
test: test-config test-elementwise-add test-gemm test-cuda-baseline-elementwise-add test-benchmark-runner

test-config: $(BUILD_DIR)/test_config_parser
	@echo "=========================================="
	@echo "Running Configuration Parser Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_config_parser

$(BUILD_DIR)/test_config_parser: tests/test_config_parser.cpp $(BUILD_DIR)/libcutekernellib_config.a
	@echo "Compiling configuration parser test..."
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib_config \
		-lyaml-cpp \
		-o $@
	@echo "  ✓ Test program compiled"

test-elementwise-add: $(BUILD_DIR)/test_elementwise_add
	@echo "=========================================="
	@echo "Running Element-wise Add Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_elementwise_add

$(BUILD_DIR)/test_elementwise_add: tests/test_elementwise_add.cpp $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling element-wise add test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-o $@
	@echo "  ✓ Test program compiled"

test-gemm: $(BUILD_DIR)/test_gemm
	@echo "=========================================="
	@echo "Running GEMM Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_gemm

$(BUILD_DIR)/test_gemm: tests/test_gemm.cpp $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling GEMM test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-o $@
	@echo "  ✓ Test program compiled"

test-gemm-performance: $(BUILD_DIR)/test_gemm_performance
	@echo "=========================================="
	@echo "Running GEMM Performance Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_gemm_performance

$(BUILD_DIR)/test_gemm_performance: tests/test_gemm_performance.cpp $(BUILD_DIR)/libcutekernellib.a
	@echo "Compiling GEMM performance test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		$< \
		-L$(BUILD_DIR) -lcutekernellib \
		-o $@
	@echo "  ✓ Performance test program compiled"

test-cuda-baseline-elementwise-add: $(BUILD_DIR)/test_cuda_baseline_elementwise_add
	@echo "=========================================="
	@echo "Running CUDA Baseline Element-wise Add Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_cuda_baseline_elementwise_add

$(BUILD_DIR)/test_cuda_baseline_elementwise_add: tests/test_cuda_baseline_elementwise_add.cpp $(BUILD_DIR)/libcutekernellib_baselines.a
	@echo "Compiling CUDA baseline element-wise add test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		-I$(BASELINES_DIR)/cuda \
		$< \
		-L$(BUILD_DIR) -lcutekernellib_baselines \
		-o $@
	@echo "  ✓ Test program compiled"

test-benchmark-runner: $(BUILD_DIR)/test_benchmark_runner
	@echo "=========================================="
	@echo "Running Benchmark Runner Test"
	@echo "=========================================="
	@$(BUILD_DIR)/test_benchmark_runner

$(BUILD_DIR)/benchmark_runner.o: $(BENCHMARKS_DIR)/benchmark_runner.cpp
	@echo "Compiling benchmark runner..."
	@$(MKDIR) $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@
	@echo "  ✓ Benchmark runner compiled"

$(BUILD_DIR)/test_benchmark_runner: $(BENCHMARKS_DIR)/test_benchmark_runner.cu $(BUILD_DIR)/benchmark_runner.o
	@echo "Compiling benchmark runner test..."
	@$(MKDIR) $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) \
		-I$(BENCHMARKS_DIR) \
		$< $(BUILD_DIR)/benchmark_runner.o \
		-o $@
	@echo "  ✓ Test program compiled"

# ============================================================================
# Benchmark Targets
# ============================================================================
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
		$(MAKE) bench-$$op || exit 1; \
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
# Profiling Targets - Nsight Compute Integration
# ============================================================================

# Check if ncu is available
NCU := $(shell which ncu 2>/dev/null)
ifeq ($(NCU),)
    NCU := $(shell which nv-nsight-cu-cli 2>/dev/null)
endif

tune-elementwise-add: $(BUILD_DIR)/bench_elementwise_add
	@echo "=========================================="
	@echo "Profiling Element-wise Add with Nsight Compute"
	@echo "=========================================="
	@if [ -z "$(NCU)" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found"; \
		echo ""; \
		echo "  Nsight Compute is required for profiling."; \
		echo "  Please install it from:"; \
		echo "    https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		echo "  On Linux, ncu is typically installed with CUDA toolkit."; \
		echo "  Try adding CUDA bin directory to PATH:"; \
		echo "    export PATH=/usr/local/cuda/bin:\$$PATH"; \
		echo ""; \
		exit 1; \
	fi
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
	@if [ -z "$(NCU)" ]; then \
		echo ""; \
		echo "  ✗ ERROR: Nsight Compute (ncu) not found"; \
		echo ""; \
		echo "  Nsight Compute is required for profiling."; \
		echo "  Please install it from:"; \
		echo "    https://developer.nvidia.com/nsight-compute"; \
		echo ""; \
		echo "  On Linux, ncu is typically installed with CUDA toolkit."; \
		echo "  Try adding CUDA bin directory to PATH:"; \
		echo "    export PATH=/usr/local/cuda/bin:\$PATH"; \
		echo ""; \
		exit 1; \
	fi
	@$(MKDIR) $(PROFILING_DIR)
	@echo ""
	@echo "Running ncu profiler (this may take several minutes)..."
	@echo "Output will be saved to: $(PROFILING_DIR)/gemm_profile.ncu-rep"
	@echo ""
	@$(NCU) --set full \
		--export $(PROFILING_DIR)/gemm_profile \
		--force-overwrite \
		--target-processes all \
		$(BUILD_DIR)/bench_gemm || \
		{ echo ""; \
		  echo "  ⚠ Warning: Profiling completed with errors (may be due to driver compatibility)"; \
		  echo "  Profile data was still collected and saved."; \
		  echo ""; \
		}
	@echo ""
	@echo "Generating human-readable summary..."
	@if [ -f "$(PROFILING_DIR)/gemm_profile.ncu-rep" ]; then \
		$(NCU) --import $(PROFILING_DIR)/gemm_profile.ncu-rep \
			--page raw \
			--csv > $(PROFILING_DIR)/gemm_summary.txt 2>&1 || \
		$(NCU) --import $(PROFILING_DIR)/gemm_profile.ncu-rep \
			> $(PROFILING_DIR)/gemm_summary.txt 2>&1 || \
		echo "Summary generation skipped (ncu-rep file may be incomplete)"; \
	else \
		echo "  ⚠ Warning: Profile report not found, skipping summary generation"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "Profiling Complete!"
	@echo "=========================================="
	@if [ -f "$(PROFILING_DIR)/gemm_profile.ncu-rep" ]; then \
		echo "Results saved to:"; \
		echo "  - NCU Report:  $(PROFILING_DIR)/gemm_profile.ncu-rep"; \
		if [ -f "$(PROFILING_DIR)/gemm_summary.txt" ]; then \
			echo "  - Summary:     $(PROFILING_DIR)/gemm_summary.txt"; \
		fi; \
		echo ""; \
		echo "To view detailed report in Nsight Compute GUI:"; \
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
		$(MAKE) tune-$$op || exit 1; \
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
