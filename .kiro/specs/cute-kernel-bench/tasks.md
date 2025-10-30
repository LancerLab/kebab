# Implementation Plan

## Development Philosophy

**CRITICAL REQUIREMENTS**:
1. **No Workarounds**: Every implementation must be genuine and functional - no placeholder code, no fake functionality
2. **Verification Required**: Each task MUST be verified by actually running `make <target>` commands - not just compilation
3. **Iterative Development**: Complete one full operator per iteration (framework + operator + baseline + benchmark + profiling)
4. **Maximum Performance**: Use Tensor Cores (mma.sync/MMA atoms), async copy, TMA, pipelining - target ≥90% of vendor library performance

## Iteration 1: Framework + Element-wise Add Operator ✓ COMPLETE

- [x] 1. Set up project structure and build system foundation
  - Create directory structure: include/cutekernellib/operators/, src/operators/, src/config/, baselines/cuda/, benchmarks/, third_party/, build/, bench_results/, profiling/
  - Create root-level config.yaml with build, benchmark, profiling, and operator configuration sections
  - Create README.md with project overview
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 4.1_

- [x] 2. Implement Makefile with GPU detection and core targets
- [x] 2.1 Create Makefile with GPU architecture detection
  - Implement CUDA_ARCH detection using nvidia-smi shell command in Makefile
  - Add OS detection (Linux/Windows) using Make's shell function
  - Define NVCC_FLAGS with architecture-specific flags (-arch=$(CUDA_ARCH))
  - Add error handling for GPU detection failures with descriptive messages
  - **Verification**: Run `make` to test GPU detection, verify CUDA_ARCH variable is set correctly
  - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 10.3, 9.6_

- [x] 2.2 Implement setup target for CuTe dependency installation
  - Add setup target to clone CUTLASS repository to third_party/cute/
  - Initialize git submodules recursively
  - Add success/failure logging with troubleshooting hints
  - **Verification**: Run `make setup` on clean environment, verify third_party/cute/ exists with CuTe headers
  - _Requirements: 2.3, 2.4, 10.1, 10.4, 9.6, 9.7_

- [x] 2.3 Implement configuration parser with yaml-cpp
  - Write include/cutekernellib/config/config_parser.h with ConfigParser singleton class
  - Implement src/config/config_parser.cpp with yaml-cpp to parse config.yaml
  - Add methods: getBuildMode(), getWarmupRuns(), getMeasurementRuns(), getBatchSizes(), getProfilingMetrics()
  - Add error handling for missing config file and invalid YAML syntax
  - **Verification**: Create test program that loads config.yaml and prints parsed values
  - _Requirements: 4.2, 4.3, 4.4, 4.5, 9.6_

- [x] 2.4 Add build target to compile configuration parser
  - Implement build target in Makefile to compile config_parser.cpp
  - Link yaml-cpp library (build from source if needed)
  - Generate libcutekernellib_config.a
  - **Verification**: Run `make build`, verify library is created in build/ directory
  - _Requirements: 2.5, 2.6, 4.5, 9.6_

- [x] 3. Implement element-wise add operator (CuTe version)
- [x] 3.1 Create element-wise add operator with vectorized memory access
  - Write include/cutekernellib/operators/elementwise_add.h with public API
  - Implement src/operators/elementwise_add.cu using CuTe for vectorized loads/stores
  - Use vector types (float4, half2) for memory coalescing
  - Add CUDA_CHECK macro for error handling
  - Add explicit template instantiations for float and half
  - **Verification**: Compile operator, run simple host program to verify correctness against CPU reference
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 10.2, 9.6, 9.7_

- [x] 3.2 Update Makefile to compile element-wise add operator
  - Add elementwise_add to OPERATORS list in Makefile
  - Update build target to compile elementwise_add.cu and link into libcutekernellib.a
  - **Verification**: Run `make build`, verify no compilation errors, library includes operator
  - _Requirements: 2.5, 9.6_

- [x] 4. Implement element-wise add baseline (optimized CUDA)
- [x] 4.1 Create hand-optimized CUDA baseline for element-wise add
  - Implement baselines/cuda/cuda_elementwise_add.cu with vectorized memory access
  - Use same API signature as CuTe version for fair comparison
  - Optimize with: vector loads/stores, grid-stride loops, proper block sizing
  - **Verification**: Compile and run standalone test, verify correctness and performance
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.6, 9.7_

- [x] 4.2 Update Makefile to compile CUDA baseline
  - Add target to compile cuda_elementwise_add.cu
  - **Verification**: Run `make build`, verify baseline compiles successfully
  - _Requirements: 5.1, 9.6_

- [x] 5. Implement benchmarking framework
- [x] 5.1 Create benchmark runner infrastructure with CUDA events
  - Write benchmarks/benchmark_runner.h with BenchmarkRunner class
  - Implement measureLatency() using cudaEvent timing with warmup and measurement phases
  - Implement calculateThroughput() for bandwidth calculation (GB/s)
  - Define BenchmarkResult struct for storing results
  - **Verification**: Create simple test that times a dummy kernel, verify timing is reasonable
  - _Requirements: 6.2, 6.3, 6.4, 6.5, 9.6_

- [x] 5.2 Create element-wise add benchmark driver
  - Implement benchmarks/bench_elementwise_add.cu using BenchmarkRunner
  - Benchmark both CuTe and CUDA baseline across batch sizes from config.yaml
  - Compute speedup ratio and bandwidth utilization
  - Output results to CSV: bench_results/elementwise_add_results.csv
  - **Verification**: Run benchmark manually, verify CSV is generated with valid data
  - _Requirements: 6.1, 6.6, 6.7, 7.4, 7.5, 9.6, 9.7_

- [x] 5.3 Add bench-elementwise_add Makefile target
  - Create bench-elementwise_add target to compile and run benchmark
  - Parse config.yaml for benchmark parameters
  - **Verification**: Run `make bench-elementwise_add`, verify benchmark executes and generates results
  - _Requirements: 6.1, 9.6, 9.7_

- [x] 6. Implement profiling system with Nsight Compute
- [x] 6.1 Add tune-elementwise_add Makefile target
  - Create tune-elementwise_add target to run ncu profiler on operator
  - Generate ncu-rep file in profiling/ directory
  - Extract summary metrics to text file
  - Parse config.yaml for ncu metric sets
  - **Verification**: Run `make tune-elementwise_add`, verify .ncu-rep and summary.txt are generated
  - _Requirements: 7A.1, 7A.2, 7A.3, 7A.4, 7A.5, 9.6, 9.7_

- [x] 6.2 Add tune-all Makefile target
  - Create tune-all target to profile all operators sequentially
  - **Verification**: Run `make tune-all`, verify profiling files for all operators
  - _Requirements: 7A.1, 9.6_

- [x] 7. Create report generation for element-wise add
- [x] 7.1 Implement Python script for benchmark report generation
  - Write scripts/generate_report.py to parse CSV files
  - Generate Markdown table with operator, variant, batch size, latency, throughput, speedup
  - Output to bench_results/summary.md
  - **Verification**: Run script manually, verify summary.md is well-formatted
  - _Requirements: 7.2, 7.3, 9.6_

- [x] 7.2 Add bench-all Makefile target for iteration 1
  - Create bench-all target to run all benchmarks and generate summary
  - **Verification**: Run `make bench-all`, verify summary.md is generated with complete data
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.6, 9.7_

- [x] 8. Documentation for iteration 1
- [x] 8.1 Write README.md with quickstart guide
  - Add project overview, technology stack, and architecture summary
  - Include quickstart: `make setup`, `make build`, `make bench-elementwise_add`
  - Add troubleshooting section for GPU detection and dependency issues
  - **Verification**: Follow README on clean environment, verify all steps work
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 10.3, 10.4, 9.6, 9.7_

- [x] 8.2 Create developer guide for adding operators
  - Document step-by-step process for adding new operators
  - Include code templates for operator, baseline, and benchmark
  - Explain Makefile integration and config.yaml extension
  - **Verification**: Review guide for completeness and clarity
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

## Iteration 2: GEMM with Tensor Cores ✓ COMPLETE

- [x] 9. Implement GEMM operator with Tensor Cores (CuTe version)
- [x] 9.1 Create GEMM operator using CuTe MMA atoms
  - Write include/cutekernellib/operators/gemm.h with public API
  - Implement src/operators/gemm.cu using CuTe MMA atoms (SM80_16x8x16_F32F16F16F32_TN for Ampere)
  - Use TiledMMA for thread block organization
  - Implement software pipelining with async copy (cp.async) for data loading
  - Use shared memory tiling for A and B matrices
  - Target ≥90% of cuBLAS performance
  - **Verification**: Run against cuBLAS, verify correctness and performance within 10% of cuBLAS
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 10.2, 9.6, 9.7_

- [x] 9.2 Update Makefile to compile GEMM operator
  - Add gemm to OPERATORS list
  - Update build target to compile gemm.cu
  - **Verification**: Run `make build`, verify GEMM compiles without errors
  - _Requirements: 2.5, 9.6_

- [x] 10. Implement GEMM baseline with Tensor Cores (optimized CUDA)
- [x] 10.1 Create hand-optimized CUDA GEMM baseline
  - Implement baselines/cuda/cuda_gemm.cu using wmma or mma.sync instructions
  - Use shared memory tiling (e.g., 128x128 tiles)
  - Implement double buffering for overlapping compute and memory
  - Use warp-level matrix operations for Tensor Core utilization
  - **Verification**: Run against cuBLAS, verify performance is competitive
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.6, 9.7_

- [x] 10.2 Update Makefile to compile GEMM baseline
  - Add target to compile cuda_gemm.cu
  - **Verification**: Run `make build`, verify baseline compiles
  - _Requirements: 5.1, 9.6_

- [x] 11. Create GEMM benchmark and profiling
- [x] 11.1 Implement GEMM benchmark driver
  - Write benchmarks/bench_gemm.cu using BenchmarkRunner
  - Test multiple matrix sizes from config.yaml
  - Compute GFLOPS (2*M*N*K operations)
  - Compare against cuBLAS as additional reference
  - Output to bench_results/gemm_results.csv
  - **Verification**: Run `make bench-gemm`, verify results show high GFLOPS and competitive speedup
  - _Requirements: 6.1, 6.6, 6.7, 7.4, 7.5, 9.6, 9.7_

- [x] 11.2 Add bench-gemm and tune-gemm Makefile targets
  - Create bench-gemm target
  - Create tune-gemm target for ncu profiling
  - **Verification**: Run both targets, verify benchmark and profiling outputs
  - _Requirements: 6.1, 7A.1, 7A.2, 7A.3, 9.6, 9.7_

- [x] 11.3 Update bench-all and tune-all for GEMM
  - Add GEMM to bench-all and tune-all targets
  - **Verification**: Run `make bench-all` and `make tune-all`, verify GEMM is included
  - _Requirements: 7.1, 7A.1, 9.6_

- [x] 12. Analyze GEMM performance and optimize
- [x] 12.1 Profile GEMM with Nsight Compute
  - Run `make tune-gemm` to generate profiling data
  - Analyze Tensor Core utilization, memory bandwidth, occupancy
  - Identify bottlenecks (compute-bound vs memory-bound)
  - **Verification**: Review ncu report, verify Tensor Cores are utilized (check sm__inst_executed_pipe_tensor metrics)
  - _Requirements: 7A.1, 7A.2, 7A.3, 7A.4, 9.6, 9.7_

- [x] 12.2 Optimize GEMM based on profiling insights
  - Adjust tile sizes, thread block dimensions based on profiling
  - Tune pipeline stages for better compute/memory overlap
  - Optimize shared memory bank conflicts if present
  - **Verification**: Re-run benchmark and profiling, verify performance improvement
  - _Requirements: 1.5, 1.6, 1.7, 9.6, 9.7_

- [x] 13. Update documentation for GEMM
- [x] 13.1 Update README with GEMM examples
  - Add GEMM usage examples and performance expectations
  - **Verification**: Review README for accuracy
  - _Requirements: 9.1, 9.2, 9.6_

- [x] 13.2 Document Tensor Core usage patterns
  - Add section in developer guide explaining MMA atom usage
  - Include code examples for Tensor Core kernels
  - **Verification**: Review documentation for completeness
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

## Iteration 3: Additional Operator (Convolution or Reduction)

This iteration is optional and can be pursued if additional operators are needed. The framework is complete and ready for extension.

- [ ] 14. Implement third operator (choose one: Conv2D or Reduction)
- [ ] 14.1 Create operator with CuTe (using appropriate hardware features)
  - Write header and implementation in include/ and src/operators/
  - Use Tensor Cores if applicable (e.g., for Conv2D)
  - Use warp-level primitives for reductions
  - **Verification**: Run correctness tests against reference implementation
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 10.2, 9.6, 9.7_

- [ ] 14.2 Create optimized CUDA baseline
  - Implement in baselines/cuda/
  - Use appropriate optimization techniques for operator type
  - **Verification**: Verify correctness and performance
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.6, 9.7_

- [ ] 14.3 Create benchmark and profiling integration
  - Implement benchmark driver in benchmarks/
  - Add bench-<op> and tune-<op> Makefile targets
  - Update bench-all and tune-all
  - **Verification**: Run `make bench-<op>` and `make tune-<op>`, verify outputs
  - _Requirements: 6.1, 6.6, 6.7, 7.1, 7A.1, 7A.2, 7A.3, 9.6, 9.7_

- [ ] 14.4 Optimize based on profiling
  - Profile with ncu, analyze bottlenecks
  - Optimize implementation
  - **Verification**: Verify performance improvement after optimization
  - _Requirements: 1.5, 1.6, 1.7, 7A.1, 7A.2, 7A.3, 7A.4, 9.6, 9.7_

## Project Completion and Polish

- [ ] 15. Final integration and validation
- [ ] 15.1 Run complete pipeline on clean environment
  - Execute: `make clean`, `make setup`, `make build`, `make bench-all`, `make tune-all`
  - Verify all operators compile, benchmark, and profile successfully
  - Review bench_results/summary.md for completeness and performance targets
  - **Verification**: Complete pipeline runs without errors, all operators meet performance targets
  - _Requirements: All requirements, 9.6, 9.7_

- [ ] 15.2 Final documentation review
  - Verify README quickstart works for new users
  - Ensure developer guide covers all implemented operators
  - Check all profiling and benchmarking documentation
  - **Verification**: Have another developer follow documentation to add a new operator
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

- [ ]* 15.3 Create Doxygen API documentation
  - Set up Doxyfile configuration
  - Add make docs target
  - Generate HTML documentation
  - _Requirements: 2.7_
