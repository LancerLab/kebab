# CuTeKernelLib Task Roadmap

## Overview

This document provides a high-level overview of the implementation roadmap. Detailed tasks are in `.kiro/specs/cute-kernel-bench/tasks.md`.

## Current Status: Phase 1 Complete ✅

- ✅ Framework and build system
- ✅ Element-wise add operator (CuTe implementation)
- ✅ GEMM basic implementation (clean, correct)
- ✅ Benchmarking and profiling infrastructure
- ✅ Performance baseline established

**Current GEMM Performance**:
- FP32: 12,327 GFLOPS (54% of cuBLAS)
- FP16: 17,889 GFLOPS (8% of cuBLAS, no Tensor Cores)

## Phase 2: GEMM Performance Optimization (CRITICAL)

**Must complete before moving to other operators**

### Phase 2A: Tensor Cores (WGMMA) - 6 tasks
**Target**: 70-80% of cuBLAS  
**Expected**: 10-20x speedup for FP16  
**Time**: 1-2 weeks

Key tasks:
1. Study CUTLASS WGMMA examples
2. Implement WGMMA atoms for Hopper
3. Optimize shared memory layouts (swizzling)
4. Implement WGMMA compute loop
5. Tune tile sizes
6. Benchmark and validate

### Phase 2B: Async Copy (TMA) - 6 tasks
**Target**: 80-90% of cuBLAS  
**Expected**: 1.5-2x improvement over Phase 2A  
**Time**: 1-2 weeks

Key tasks:
1. Study CUTLASS TMA examples
2. Create TMA descriptors
3. Implement TMA async loads
4. Implement multi-stage pipeline (3 stages)
5. Optimize TMA parameters
6. Benchmark and validate

### Phase 2C: Thread Block Clusters - 5 tasks
**Target**: 90-95% of cuBLAS  
**Expected**: 1.2-1.3x improvement over Phase 2B  
**Time**: 1 week

Key tasks:
1. Study CUTLASS cluster examples
2. Implement cluster launch
3. Implement distributed shared memory
4. Optimize for larger effective memory
5. Benchmark and validate

### Phase 2D: Final Optimization & FP8 - 4 tasks
**Target**: 95%+ of cuBLAS  
**Expected**: 2x additional throughput with FP8  
**Time**: 1 week

Key tasks:
1. Implement FP8 Tensor Core support
2. Fine-tune all parameters
3. Implement epilogue fusion
4. Final benchmark and documentation

**Total Phase 2 Time**: 4-6 weeks

## Phase 3: Convolution Operator

**Start after Phase 2 complete**

### Phase 3A: Basic Implementation - 5 tasks
**Target**: 50-60% of cuDNN  
**Time**: 1-2 weeks

### Phase 3B: Winograd Algorithm - 4 tasks
**Target**: 70-80% of cuDNN (3x3)  
**Time**: 1-2 weeks

### Phase 3C: Direct Conv Optimization - 4 tasks
**Target**: 80-90% of cuDNN  
**Time**: 1-2 weeks

**Total Phase 3 Time**: 3-6 weeks

## Phase 4: Reduction Operators

**Start after Phase 2 complete** (can overlap with Phase 3)

### Phase 4A: Basic Warp-level - 4 tasks
**Target**: 60-70% of CUB  
**Time**: 1 week

### Phase 4B: Multi-stage Optimization - 3 tasks
**Target**: 80-90% of CUB  
**Time**: 1 week

**Total Phase 4 Time**: 2 weeks

## Phase 5: Additional Operators

**Start after Phase 2 complete**

### 5.1 Softmax - 2 tasks
**Target**: 80-90% of vendor library  
**Time**: 3-5 days

### 5.2 LayerNorm - 2 tasks
**Target**: 85-95% of vendor library  
**Time**: 3-5 days

### 5.3 Batched GEMM - 1 task
**Target**: Same per-GEMM performance  
**Time**: 3-5 days

### 5.4 Grouped GEMM - 1 task
**Target**: 80-90% of cuBLAS  
**Time**: 5-7 days

**Total Phase 5 Time**: 2-3 weeks

## Phase 6: Final Integration

### 6.1 Performance Validation - 2 tasks
**Time**: 3-5 days

### 6.2 Documentation - 3 tasks
**Time**: 5-7 days

**Total Phase 6 Time**: 1-2 weeks

## Total Project Timeline

**Minimum**: 12-15 weeks  
**Expected**: 15-20 weeks  
**With buffer**: 20-24 weeks (5-6 months)

## Critical Path

1. **Phase 2 (GEMM Optimization)**: MUST complete first
   - This establishes the performance baseline
   - Techniques learned apply to other operators
   - Most complex optimization work

2. **Phase 3 (Convolution)**: Can start after Phase 2
   - Reuses GEMM kernels
   - Second most important operator

3. **Phase 4 (Reduction)**: Can overlap with Phase 3
   - Independent from GEMM/Conv
   - Needed for Phase 5 operators

4. **Phase 5 (Additional)**: Depends on Phase 2 and 4
   - Builds on previous work
   - Relatively quick to implement

5. **Phase 6 (Integration)**: Final polish
   - Documentation and validation
   - No new features

## Success Metrics

### Performance (Must Achieve)
- GEMM: ≥95% of cuBLAS
- Conv2D: ≥90% of cuDNN
- Reduction: ≥85% of CUB
- Element-wise: ≥95% of bandwidth
- Others: ≥85% of vendor libraries

### Code Quality (Must Maintain)
- Pure CuTe in `src/` (no raw CUDA)
- All hardware features utilized
- 100% correctness
- Comprehensive tests

### Documentation (Must Complete)
- API documentation (Doxygen)
- Performance guide
- Developer guide
- Usage examples

## Risk Mitigation

### High Risk: Phase 2 Performance
**Risk**: May not reach 95% of cuBLAS  
**Mitigation**: 
- Study CUTLASS implementation in detail
- Iterate on optimizations
- Profile extensively
- Consult NVIDIA documentation

### Medium Risk: TMA Complexity
**Risk**: TMA may be difficult to implement correctly  
**Mitigation**:
- Start with simple examples
- Validate each step
- Use CUTLASS as reference

### Low Risk: Other Operators
**Risk**: May not reach performance targets  
**Mitigation**:
- Reuse GEMM techniques
- Follow established patterns
- Iterate based on profiling

## Next Steps

1. **Immediate**: Begin Phase 2A (WGMMA)
   - Study CUTLASS examples (2-3 hours)
   - Implement first WGMMA kernel (1-2 days)
   - Validate correctness (1 day)

2. **This Week**: Complete Phase 2A tasks 1-3
   - WGMMA atom implementation
   - Shared memory layouts
   - Initial benchmarks

3. **Next Week**: Complete Phase 2A tasks 4-6
   - Compute loop optimization
   - Tile size tuning
   - Final Phase 2A validation

## Resources

### CUTLASS Examples
- `cutlass/examples/cute/tutorial/wgmma_gemm.cu`
- `cutlass/examples/cute/tutorial/sgemm_sm90.cu`
- `cutlass/examples/cute/tutorial/tma_load.cu`
- `cutlass/examples/cute/tutorial/cluster_gemm.cu`

### Documentation
- [CuTe Quickstart](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [Hopper Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [CUTLASS 3.x Design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cutlass_3x_design.md)

### Internal Docs
- `docs/PERFORMANCE_BASELINE.md` - Current performance
- `docs/CUTE_GEMM_IMPLEMENTATION.md` - Implementation guide
- `docs/IMPLEMENTATION_STATUS.md` - Feature tracking
- `src/operators/README_GEMM_CUTE.md` - CuTe features

## Status Tracking

Track progress in:
- `.kiro/specs/cute-kernel-bench/tasks.md` - Detailed task list
- `docs/IMPLEMENTATION_STATUS.md` - Feature matrix
- `docs/PERFORMANCE_*.md` - Performance records

Update after each phase completion.
