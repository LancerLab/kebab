# Requirements Document

## Introduction

CuTeKernelLib (kebab - cute-kernel-bench) is a high-performance kernel library leveraging NVIDIA CUTLASS CuTe for implementing AI and scientific computing operators. The library provides a modular, benchmarking-focused framework that compares CuTe template implementations against optimized CUDA baselines, enabling performance verification and operator development with minimal manual configuration.

## Glossary

- **CuTeKernelLib**: The high-performance kernel library system being specified (also known as "kebab")
- **CuTe**: NVIDIA CUTLASS CuTe, a C++ template header-only library for implementing GPU operators
- **Operator**: A computational kernel (e.g., GEMM, FFT, LU decomposition) implemented using CuTe templates
- **Baseline**: Hand-optimized CUDA reference implementation for performance comparison
- **Build System**: The GNU Make-based compilation and automation framework
- **Configuration System**: YAML-based centralized parameter management via config.yaml
- **Benchmark Framework**: Custom timing and comparison system using CUDA events
- **Setup Target**: Makefile target that installs dependencies and prepares the development environment
- **GPU Architecture**: NVIDIA GPU compute capability (e.g., sm_80 for Ampere)
- **Host System**: The development machine running the build and benchmark processes

## Requirements

### Requirement 1: Operator Implementation Framework

**User Story:** As a library developer, I want to implement GPU operators using CuTe templates with a clear file structure, so that I can create modular, maintainable, and debuggable kernel implementations.

#### Acceptance Criteria

1. THE CuTeKernelLib SHALL organize operator header files in the directory `include/cutekernellib/operators/` with one header file per operator
2. THE CuTeKernelLib SHALL organize operator source files in the directory `src/operators/` with one .cu file per operator containing CuTe template implementations
3. THE CuTeKernelLib SHALL provide a self-contained implementation for each Operator where the source file includes CuTe headers, defines templates, and exposes a C++ API function
4. THE CuTeKernelLib SHALL implement thread-safe and deterministic Operator execution with CUDA runtime error checking
5. THE CuTeKernelLib SHALL leverage maximum hardware capabilities including Tensor Cores via mma.sync instructions and CuTe MMA atoms for compute-intensive operators
6. THE CuTeKernelLib SHALL utilize advanced memory features including asynchronous copy, TMA, and pipeline mechanisms for data movement where applicable
7. THE CuTeKernelLib SHALL target SOTA performance levels comparable to vendor-optimized libraries

### Requirement 2: Build System Architecture

**User Story:** As a developer, I want a Makefile-based build system that handles all compilation and setup tasks, so that I can build the library without writing shell scripts.

#### Acceptance Criteria

1. THE Build System SHALL provide a root-level Makefile as the single entry point for all project interactions
2. THE Build System SHALL implement all automation logic using Make's built-in functions and variables without external shell scripts
3. WHEN the developer invokes the setup target, THE Build System SHALL detect the Host System operating system using Make variables
4. WHEN the developer invokes the setup target, THE Build System SHALL clone the CuTe repository to `third_party/cute/` and fetch required submodules
5. WHEN the developer invokes the build target, THE Build System SHALL compile all Operator source files using NVCC and generate static library `libcutekernellib.a`
6. THE Build System SHALL support debug and release compilation modes as specified in the Configuration System
7. THE Build System SHALL provide clean, docs, and test targets for maintenance and verification

### Requirement 3: GPU Architecture Detection

**User Story:** As a developer, I want automatic GPU architecture detection during build, so that the library compiles with optimal architecture-specific flags without manual configuration.

#### Acceptance Criteria

1. WHEN the Build System executes compilation, THE Build System SHALL detect GPU compute capability using nvidia-smi query commands
2. THE Build System SHALL set the CUDA_ARCH Make variable based on detected compute capability
3. THE Build System SHALL pass architecture-specific compilation flags to NVCC using the detected CUDA_ARCH value
4. IF GPU detection fails, THEN THE Build System SHALL log an error message and halt compilation with a non-zero exit code

### Requirement 4: Configuration Management

**User Story:** As a user, I want to configure all library parameters through a single YAML file, so that I can tune benchmarks and builds without modifying code or Makefiles.

#### Acceptance Criteria

1. THE Configuration System SHALL provide a root-level config.yaml file as the single source of configuration parameters
2. THE Configuration System SHALL support benchmark parameters including warmup_runs, repeats, and batch_sizes arrays
3. THE Configuration System SHALL support build parameters including compilation mode and optimization flags
4. WHEN the Build System processes configuration, THE Configuration System SHALL parse config.yaml using the yaml-cpp library
5. THE Build System SHALL propagate Configuration System values to compilation via compiler definition flags

### Requirement 5: Baseline Reference Implementations

**User Story:** As a performance engineer, I want optimized CUDA baseline implementations for each operator, so that I can verify CuTe implementations achieve competitive performance.

#### Acceptance Criteria

1. THE CuTeKernelLib SHALL organize Baseline implementations in the directory `baselines/cuda/` with one .cu file per operator
2. THE CuTeKernelLib SHALL implement each Baseline using hand-optimized CUDA techniques including tiling, shared memory, and warp-level primitives
3. THE CuTeKernelLib SHALL ensure each Baseline processes identical input shapes and data types as the corresponding CuTe Operator
4. THE CuTeKernelLib SHALL synchronize host and device execution for fair timing comparison between Baseline and Operator implementations

### Requirement 6: Benchmarking Framework

**User Story:** As a performance engineer, I want to benchmark CuTe operators against CUDA baselines with precise timing, so that I can quantify performance differences and identify optimization opportunities.

#### Acceptance Criteria

1. WHEN the developer invokes a bench-<op_name> target, THE Benchmark Framework SHALL compile and execute the specified Operator benchmark
2. THE Benchmark Framework SHALL measure execution latency using CUDA events for microsecond precision
3. THE Benchmark Framework SHALL execute warmup iterations as specified in the Configuration System before measurement
4. THE Benchmark Framework SHALL execute measurement iterations as specified in the Configuration System and compute average latency
5. THE Benchmark Framework SHALL calculate throughput in GFLOPS for both CuTe Operator and Baseline implementations
6. THE Benchmark Framework SHALL compute speedup ratio between CuTe Operator and Baseline for each test configuration
7. THE Benchmark Framework SHALL test multiple batch sizes as specified in the Configuration System batch_sizes array

### Requirement 7: Aggregate Benchmark Reporting

**User Story:** As a project maintainer, I want to run all benchmarks and generate a summary report, so that I can assess overall library performance at a glance.

#### Acceptance Criteria

1. WHEN the developer invokes the bench-all target, THE Benchmark Framework SHALL execute benchmarks for all implemented Operators sequentially
2. THE Benchmark Framework SHALL aggregate results into a Markdown summary file at `bench_results/summary.md`
3. THE Benchmark Framework SHALL include columns for Operator name, Variant (CuTe or CUDA), Batch Size, Latency in milliseconds, Throughput in GFLOPS, and Speedup Ratio in the summary
4. THE Benchmark Framework SHALL log raw benchmark data to CSV files for reproducibility and post-processing
5. THE Benchmark Framework SHALL auto-detect GPU architecture via the Build System for benchmark execution

### Requirement 7A: Performance Profiling and Tuning

**User Story:** As a performance engineer, I want to profile operators using NVIDIA Nsight Compute, so that I can analyze kernel performance metrics and identify optimization opportunities.

#### Acceptance Criteria

1. WHEN the developer invokes a tune-<op_name> target, THE Build System SHALL execute the specified Operator using ncu profiler
2. THE Build System SHALL generate detailed profiling reports including metrics for compute throughput, memory bandwidth, and occupancy
3. THE Build System SHALL output ncu analysis files to a dedicated profiling/ directory
4. THE Build System SHALL generate a human-readable summary of key performance bottlenecks
5. THE Build System SHALL support profiling with custom ncu metric sets specified in config.yaml

### Requirement 8: Modular File Organization

**User Story:** As a developer, I want a hierarchical and modular file structure, so that I can navigate the codebase efficiently and understand component boundaries.

#### Acceptance Criteria

1. THE CuTeKernelLib SHALL organize the project with root-level Makefile, config.yaml, and README.md files
2. THE CuTeKernelLib SHALL provide a benchmarks/ directory containing benchmark driver files with one .cu file per Operator
3. THE CuTeKernelLib SHALL maintain third_party/cute/ directory for CuTe dependency installation
4. THE CuTeKernelLib SHALL generate build artifacts in a dedicated build/ directory separate from source files
5. THE CuTeKernelLib SHALL output benchmark results to a dedicated bench_results/ directory

### Requirement 9: Developer Extensibility and Verification

**User Story:** As a new contributor, I want clear guidance on adding new operators with strict verification requirements, so that I can extend the library following established patterns without workarounds.

#### Acceptance Criteria

1. THE CuTeKernelLib SHALL provide documentation specifying the steps to create a new Operator source file in src/operators/
2. THE CuTeKernelLib SHALL provide documentation specifying the steps to add the new Operator to the Build System compilation list
3. THE CuTeKernelLib SHALL provide documentation specifying the steps to create a corresponding Baseline implementation
4. THE CuTeKernelLib SHALL provide documentation specifying the steps to create a benchmark driver in benchmarks/
5. WHERE a new Operator requires custom parameters, THE CuTeKernelLib SHALL provide documentation specifying the steps to extend config.yaml
6. WHEN implementing any task, THE CuTeKernelLib development process SHALL require actual execution of make targets to verify functionality without fallback implementations or workarounds
7. THE CuTeKernelLib development process SHALL reject placeholder implementations and require genuine functional code that passes real execution tests

### Requirement 10: Error Handling and Logging

**User Story:** As a user, I want clear error messages and logs during setup, build, and benchmark execution, so that I can troubleshoot issues efficiently.

#### Acceptance Criteria

1. WHEN the Build System encounters a dependency installation failure, THE Build System SHALL log a descriptive error message and halt with a non-zero exit code
2. WHEN an Operator encounters a CUDA runtime error, THE CuTeKernelLib SHALL check the error code and report the error type and location
3. WHEN GPU detection fails, THE Build System SHALL log the failure reason and provide troubleshooting guidance
4. THE Build System SHALL output success messages upon successful completion of setup, build, and benchmark targets
5. THE Benchmark Framework SHALL log benchmark progress including current Operator and test configuration during execution
