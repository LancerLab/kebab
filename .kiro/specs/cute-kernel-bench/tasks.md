# Implementation Plan

## Development Philosophy

**CRITICAL REQUIREMENTS**:
1. **Pure CuTe**: All implementations in `src/` must use CuTe abstractions - no raw CUDA types
2. **Hardware Features**: Must utilize latest GPU features (Tensor Cores, TMA, async copy, clusters)
3. **Verification Required**: Each task MUST be verified by running actual benchmarks and tests
4. **Performance Target**: ≥90% of vendor library (cuBLAS, cuDNN) performance
5. **No Placeholders**: Every implementation must be genuine and functional

## Phase 1: Foundation & Basic Implementations ✅ COMPLETE

- [x] 1. Framework setup, element-wise add operator, benchmarking, profiling
  - Status: All tasks complete, verified working

- [x] 2. GEMM Basic Implementation
  - Status: Clean implementation, 12,327 GFLOPS (54% of cuBLAS for FP32)
  - Baseline Performance Recorded: See `docs/PERFORMANCE_BASELINE.md`

---

## Phase 2: GEMM Performance Optimization (CRITICAL - Must Complete Before Other Operators)

- [-] 3. GEMM Phase 2A: Tensor Core Integration (WGMMA)
  - Goal: Achieve 70-80% of cuBLAS performance using Hopper WGMMA instructions
  - [x] 3.1 Study CUTLASS WGMMA examples
    - Read `cutlass/examples/cute/tutorial/wgmma_gemm.cu`
    - Read `cutlass/examples/cute/tutorial/sgemm_sm90.cu`
    - Understand SM90 WGMMA atom structure
    - Document key concepts in `docs/WGMMA_NOTES.md`
    - Verification: Create summary document with code snippets
    - Time: 2-3 hours study
  - [ ] 3.2 Implement WGMMA atom selection for FP16
    - Define MMA atom: `SM90_64x256x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>`
    - Create TiledMMA with appropriate layout: `TiledMMA<MMA_Atom, Layout<Shape<_2,_1,_1>>>`
    - Add to `src/operators/gemm_wgmma.cu` (new file)
    - Verification: Compile successfully, verify atom dimensions
    - Expected: Compile-time layout validation
  - [ ] 3.3 Implement shared memory layout for WGMMA
    - Create swizzled layout for A matrix (avoid bank conflicts)
    - Create swizzled layout for B matrix
    - Use `Swizzle<3,3,3>` for 8-way swizzle
    - Allocate shared memory: `__shared__ T smem_A[BLK_M * BLK_K]`
    - Verification: Check bank conflict metrics in profiling
    - Expected: <5% bank conflicts
  - [ ] 3.4 Implement WGMMA compute loop
    - Use `gemm(tiled_mma, sA, sB, acc)` for matrix multiply
    - Partition fragments: `partition_fragment_C(tiled_mma, make_shape(BLK_M, BLK_N))`
    - Handle accumulator initialization and epilogue
    - Verification: Correctness test against cuBLAS
    - Expected: Bit-exact results (within FP16 tolerance)
  - [ ] 3.5 Optimize tile sizes for Hopper
    - Test tile sizes: 128x128, 128x256, 256x128
    - Measure occupancy and performance for each
    - Select optimal based on profiling
    - Verification: Run `make tune-gemm`, check occupancy
    - Expected: >50% occupancy, >70% of cuBLAS
  - [ ] 3.6 Benchmark WGMMA implementation
    - Run full benchmark suite: 256³ to 4096³
    - Compare against Phase 1 baseline
    - Record performance in `docs/PERFORMANCE_PHASE2A.md`
    - Verification: `make bench-gemm`
    - Expected: 10-20x speedup for FP16, 70-80% of cuBLAS

- [ ] 4. GEMM Phase 2B: Async Copy (TMA)
  - Goal: Achieve 80-90% of cuBLAS performance using Tensor Memory Accelerator
  - [ ] 4.1 Study CUTLASS TMA examples
    - Read `cutlass/examples/cute/tutorial/tma_load.cu`
    - Read `cutlass/examples/cute/tutorial/tma_store.cu`
    - Understand TMA descriptor creation
    - Document in `docs/TMA_NOTES.md`
    - Verification: Summary document with examples
    - Time: 2-3 hours study
  - [ ] 4.2 Create TMA descriptors for A and B matrices
    - Use `make_tma_copy(SM90_TMA_LOAD{}, tensor, tile_shape)`
    - Create descriptor for A: `tma_load_a`
    - Create descriptor for B: `tma_load_b`
    - Handle 2D memory layout
    - Verification: Descriptor creation succeeds
    - Expected: Valid TMA descriptors
  - [ ] 4.3 Implement TMA async load in kernel
    - Replace manual shared memory loads with TMA
    - Use `copy(tma_load_a, gA_tile, sA)`
    - Add `cp_async_wait<0>()` for synchronization
    - Verification: Correctness test
    - Expected: Same results as manual copy
  - [ ] 4.4 Implement multi-stage pipeline (3 stages)
    - Stage 0: Load next tile
    - Stage 1: Compute current tile
    - Stage 2: Store previous results
    - Use `cuda::pipeline` for synchronization
    - Verification: Profile pipeline efficiency
    - Expected: >80% compute/memory overlap
  - [ ] 4.5 Optimize TMA parameters
    - Tune number of pipeline stages (2, 3, 4)
    - Adjust tile sizes for TMA efficiency
    - Optimize barrier placement
    - Verification: Profile with `make tune-gemm`
    - Expected: 80-90% of cuBLAS
  - [ ] 4.6 Benchmark TMA implementation
    - Full benchmark suite
    - Compare against Phase 2A (WGMMA only)
    - Record in `docs/PERFORMANCE_PHASE2B.md`
    - Verification: `make bench-gemm`
    - Expected: 1.5-2x improvement over Phase 2A

- [ ] 5. GEMM Phase 2C: Thread Block Clusters
  - Goal: Achieve 90-95% of cuBLAS performance using clusters and distributed shared memory
  - [ ] 5.1 Study CUTLASS cluster examples
    - Read `cutlass/examples/cute/tutorial/cluster_gemm.cu`
    - Understand cluster launch configuration
    - Understand distributed shared memory
    - Document in `docs/CLUSTERS_NOTES.md`
    - Verification: Summary document
    - Time: 2-3 hours study
  - [ ] 5.2 Implement cluster launch configuration
    - Define cluster dimensions: `dim3 cluster_dims(2, 2, 1)`
    - Use `cudaLaunchKernelEx` for cluster launch
    - Configure cluster attributes
    - Verification: Kernel launches successfully
    - Expected: Cluster formation confirmed
  - [ ] 5.3 Implement distributed shared memory access
    - Allocate distributed shared memory
    - Access neighbor CTA's shared memory
    - Use `get_cluster_smem_ptr(neighbor_id)`
    - Verification: Correctness test
    - Expected: Correct data sharing across CTAs
  - [ ] 5.4 Optimize for larger effective shared memory
    - Use up to 228KB per cluster (vs 164KB per CTA)
    - Adjust tile sizes to utilize larger memory
    - Reduce global memory traffic
    - Verification: Profile memory bandwidth
    - Expected: >90% DRAM bandwidth utilization
  - [ ] 5.5 Benchmark cluster implementation
    - Full benchmark suite
    - Compare against Phase 2B (TMA)
    - Record in `docs/PERFORMANCE_PHASE2C.md`
    - Verification: `make bench-gemm`
    - Expected: 90-95% of cuBLAS

- [ ] 6. GEMM Phase 2D: Final Optimization & FP8 Support
  - Goal: Reach 95%+ of cuBLAS, add FP8 support for 2x additional throughput

- [ ] 2.4.1 Implement FP8 Tensor Core support
  - Add FP8 MMA atom: `SM90_64x256x64_E4M3E4M3F16_SS<>`
  - Implement FP8 input conversion
  - Handle FP16 accumulation
  - **Verification**: Correctness test with FP8 inputs
  - [ ] 6.1 Implement FP8 Tensor Core support
    - Add FP8 MMA atom: `SM90_64x256x64_E4M3E4M3F16_SS<>`
    - Implement FP8 input conversion
    - Handle FP16 accumulation
    - Verification: Correctness test with FP8 inputs
    - Expected: 2x throughput vs FP16
  - [ ] 6.2 Fine-tune all parameters
    - Optimize tile sizes per matrix size
    - Tune pipeline stages
    - Optimize register usage
    - Minimize synchronization overhead
    - Verification: Profile each optimization
    - Expected: Incremental improvements
  - [ ] 6.3 Implement epilogue fusion
    - Add support for alpha/beta scaling: `C = alpha*A*B + beta*C`
    - Fuse activation functions (ReLU, GELU)
    - Fuse bias addition
    - Verification: Correctness tests
    - Expected: No performance loss
  - [ ] 6.4 Final benchmark and documentation
    - Complete benchmark suite (FP32, FP16, FP8)
    - Create performance comparison table
    - Document all optimizations in `docs/GEMM_FINAL_REPORT.md`
    - Verification: `make bench-gemm`
    - Expected: 95%+ of cuBLAS for all precisions

---

## Phase 3: Convolution Operator (After GEMM Complete)

- [ ] 7. Conv2D Phase 3A: Basic CuTe Implementation
  - Goal: Correct implementation using CuTe, 50-60% of cuDNN
  - [ ] 7.1 Study CUTLASS convolution examples
    - Read `cutlass/examples/13_two_tensor_op_fusion/`
    - Read `cutlass/examples/cute/tutorial/` for conv patterns
    - Understand im2col vs direct convolution
    - Document in `docs/CONV_NOTES.md`
    - Verification: Summary document
    - Time: 3-4 hours study
  - [ ] 7.2 Implement Conv2D operator interface
    - Create `include/cutekernellib/operators/conv2d.h`
    - Define API: `conv2d(input, weight, output, N, C, H, W, K, R, S, stride, padding)`
    - Support NCHW and NHWC layouts
    - Verification: API compiles
    - Expected: Clean interface
  - [ ] 7.3 Implement im2col transformation using CuTe
    - Use CuTe Layout to describe im2col pattern
    - Create tensor view for input patches
    - Avoid explicit im2col materialization
    - Verification: Correctness test on small input
    - Expected: Correct patch extraction
  - [ ] 7.4 Implement Conv2D as batched GEMM
    - Reuse GEMM kernel from Phase 2
    - Map convolution to GEMM dimensions
    - Handle padding and stride in layout
    - Verification: Correctness test against cuDNN
    - Expected: Bit-exact results
  - [ ] 7.5 Optimize for common conv sizes
    - Tune for 3x3, 5x5, 7x7 kernels
    - Optimize for stride=1, stride=2
    - Handle padding efficiently
    - Verification: Benchmark common sizes
    - Expected: 50-60% of cuDNN

- [ ] 8. Conv2D Phase 3B: Winograd Algorithm
  - Goal: 70-80% of cuDNN for 3x3 convolutions
  - [ ] 8.1 Study Winograd convolution
    - Read Winograd F(2x2, 3x3) algorithm
    - Understand transform matrices
    - Study CUTLASS Winograd implementation
    - Document in `docs/WINOGRAD_NOTES.md`
    - Verification: Summary with math
    - Time: 3-4 hours study
  - [ ] 8.2 Implement Winograd transforms
    - Implement input transform: `B^T * d * B`
    - Implement filter transform: `G * g * G^T`
    - Implement output transform: `A^T * m * A`
    - Use CuTe for transform operations
    - Verification: Transform correctness
    - Expected: Correct transforms
  - [ ] 8.3 Implement Winograd Conv2D kernel
    - Apply transforms in shared memory
    - Use WGMMA for element-wise products
    - Fuse transforms with GEMM
    - Verification: Correctness test
    - Expected: Same results as direct conv
  - [ ] 8.4 Optimize Winograd implementation
    - Minimize transform overhead
    - Optimize shared memory usage
    - Use Tensor Cores for transforms
    - Verification: Profile and benchmark
    - Expected: 70-80% of cuDNN for 3x3

- [ ] 9. Conv2D Phase 3C: Direct Convolution Optimization
  - Goal: 80-90% of cuDNN using optimized direct convolution
  - [ ] 9.1 Implement optimized direct convolution
    - Use TMA for input loading
    - Use WGMMA for accumulation
    - Implement multi-stage pipeline
    - Verification: Correctness test
    - Expected: Correct results
  - [ ] 9.2 Implement specialized kernels
    - 1x1 convolution (pointwise)
    - 3x3 stride-1 (most common)
    - Depthwise convolution
    - Verification: Benchmark each type
    - Expected: Near-optimal for each
  - [ ] 9.3 Add thread block clusters
    - Use clusters for larger tile sizes
    - Share input data across CTAs
    - Reduce redundant loads
    - Verification: Profile memory traffic
    - Expected: 80-90% of cuDNN
  - [ ] 9.4 Final Conv2D optimization
    - Auto-tuning for different shapes
    - Kernel selection heuristics
    - Complete documentation
    - Verification: Full benchmark suite
    - Expected: 90%+ of cuDNN

---

## Phase 4: Reduction Operators

- [ ] 10. Reduction Phase 4A: Basic Warp-level Reduction
  - Goal: Correct implementation, 60-70% of CUB
  - [ ] 10.1 Study CuTe reduction patterns
    - Read warp-level reduction examples
    - Understand shuffle operations
    - Study CUB reduction implementation
    - Document in `docs/REDUCTION_NOTES.md`
    - Verification: Summary document
    - Time: 2-3 hours study
  - [ ] 10.2 Implement warp-level sum reduction
    - Use `__shfl_down_sync` for warp reduction
    - Implement tree reduction pattern
    - Handle partial warps
    - Verification: Correctness test
    - Expected: Correct sum
  - [ ] 10.3 Implement block-level reduction
    - Use shared memory for inter-warp reduction
    - Minimize synchronization
    - Use atomic operations for final reduction
    - Verification: Correctness test
    - Expected: Correct results
  - [ ] 10.4 Implement multiple reduction types
    - Sum, Max, Min, Product
    - Mean, Variance
    - Generic reduction with custom operator
    - Verification: Test each type
    - Expected: 60-70% of CUB

- [ ] 11. Reduction Phase 4B: Multi-stage Reduction
  - Goal: 80-90% of CUB using optimized multi-stage approach
  - [ ] 11.1 Implement two-stage reduction
    - Stage 1: Block-level partial reductions
    - Stage 2: Final reduction of partials
    - Optimize grid size selection
    - Verification: Benchmark
    - Expected: Better for large inputs
  - [ ] 11.2 Implement vectorized loads
    - Use vector types for input loading
    - Reduce in registers before shared memory
    - Minimize memory transactions
    - Verification: Profile memory bandwidth
    - Expected: >80% bandwidth utilization
  - [ ] 11.3 Add specialized kernels
    - Small input (single block)
    - Medium input (multi-block)
    - Large input (two-stage)
    - Verification: Benchmark each
    - Expected: 80-90% of CUB

---

## Phase 5: Additional Operators

- [ ] 12. Softmax (After Reduction Complete)
  - [ ] 12.1 Implement online softmax algorithm
    - Use numerically stable algorithm
    - Fuse max, exp, and sum operations
    - Use warp-level primitives
    - Verification: Correctness test
    - Expected: Numerically stable
  - [ ] 12.2 Optimize for different dimensions
    - Row-wise softmax (most common)
    - Column-wise softmax
    - Batched softmax
    - Verification: Benchmark
    - Expected: 80-90% of vendor library

- [ ] 13. LayerNorm (After Reduction Complete)
  - [ ] 13.1 Implement fused LayerNorm
    - Fuse mean, variance, and normalization
    - Use Welford's online algorithm
    - Minimize memory passes
    - Verification: Correctness test
    - Expected: Correct results
  - [ ] 13.2 Add affine transformation
    - Fuse scale and bias
    - Support different layouts
    - Verification: Benchmark
    - Expected: 85-95% of vendor library

- [ ] 14. Batched GEMM (After GEMM Phase 2 Complete)
  - [ ] 14.1 Implement strided batched GEMM
    - Extend GEMM kernel for batches
    - Use persistent threads
    - Optimize for small batch sizes
    - Verification: Correctness test
    - Expected: Same per-GEMM performance

- [ ] 15. Grouped GEMM (After Batched GEMM Complete)
  - [ ] 15.1 Implement grouped GEMM
    - Handle variable matrix sizes
    - Use dynamic parallelism or persistent threads
    - Optimize load balancing
    - Verification: Benchmark
    - Expected: 80-90% of cuBLAS

---

## Phase 6: Final Integration & Documentation

- [ ] 16. Performance Validation
  - [ ] 16.1 Complete benchmark suite
    - Run all operators on all supported sizes
    - Generate performance comparison tables
    - Create performance visualization
    - Verification: All benchmarks pass
    - Expected: All operators ≥90% of vendor libraries
  - [ ] 16.2 Regression testing
    - Create automated test suite
    - Test all operators for correctness
    - Test all precisions (FP32, FP16, FP8)
    - Verification: All tests pass
    - Expected: 100% correctness

- [ ] 17. Documentation
  - [ ] 17.1 Complete API documentation
    - Document all public APIs
    - Add usage examples
    - Create Doxygen documentation
    - Verification: `make docs` succeeds
    - Expected: Complete API docs
  - [ ] 17.2 Performance guide
    - Document performance characteristics
    - Add tuning guidelines
    - Create troubleshooting guide
    - Verification: Review for completeness
    - Expected: Comprehensive guide
  - [ ] 17.3 Developer guide
    - Document how to add new operators
    - Explain CuTe patterns used
    - Add optimization techniques
    - Verification: Follow guide to add operator
    - Expected: Clear, actionable guide

---

## Success Criteria

### Performance Targets
- **GEMM**: ≥95% of cuBLAS (all precisions)
- **Conv2D**: ≥90% of cuDNN (common sizes)
- **Reduction**: ≥85% of CUB
- **Element-wise**: ≥95% of memory bandwidth
- **Other operators**: ≥85% of vendor libraries

### Code Quality
- ✅ Pure CuTe implementations (no raw CUDA in `src/`)
- ✅ All hardware features utilized (Tensor Cores, TMA, clusters)
- ✅ Comprehensive testing (correctness + performance)
- ✅ Complete documentation

### Verification
- All tasks must be verified by running actual code
- Performance must be measured, not estimated
- Correctness must be validated against reference implementations
