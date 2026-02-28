# GEMM Kernel Optimization - Development Status

## Current Status: Analysis Complete ✅

**Date**: March 1, 2026  
**Phase**: Analysis & Planning  
**Next Phase**: Implementation

---

## What Has Been Done

### 1. Comprehensive Kernel Analysis ✅
- Analyzed all 17 existing GEMM kernel versions (v1-v17)
- Benchmarked performance on H800 GPU (FP16, M=N=K=8192)
- Identified feature patterns and performance correlations
- Created detailed feature taxonomy

### 2. Performance Insights ✅
**Current Best Performers**:
- **v15**: 335,009 GFLOPS (79.4% of cuBLAS) - non-persistent + stmatrix
- **v5**: 332,073 GFLOPS (78.7% of cuBLAS) - simple persistent ⭐
- **v14**: 298,557 GFLOPS (70.8% of cuBLAS) - persistent + stmatrix

**Key Findings**:
- v5's simplicity is a strength (beats 10 more complex kernels)
- Non-persistent helps complex kernels (+12%)
- Persistent helps simple kernels (+37%)
- Clusters hurt performance (-2%)
- Hilbert scheduling helps (+3.7%)
- stmatrix helps (+0.6%)

### 3. Optimization Opportunities Identified ✅
**4 Missing High-Value Combinations**:
1. **v18**: v5 + stmatrix → Expected: 335-340 GFLOPS
2. **v19**: v5 + hilbert → Expected: 340-345 GFLOPS
3. **v20**: v15 + warpgroup → Expected: 340-350 GFLOPS
4. **v21**: v5 + stmatrix + hilbert → Expected: 345-355 GFLOPS

### 4. Documentation Created ✅
- ✅ `analysis_gemm_features.md` - Feature taxonomy and performance data
- ✅ `GEMM_OPTIMIZATION_RECOMMENDATIONS.md` - Strategic recommendations
- ✅ `V18_IMPLEMENTATION_PLAN.md` - Step-by-step implementation guide
- ✅ `GEMM_ANALYSIS_SUMMARY.md` - Executive summary
- ✅ `KERNEL_COMPARISON_TABLE.md` - Complete feature comparison
- ✅ `DEVELOPMENT_STATUS.md` - This file

### 5. Code Cleanup ✅
- Removed 13 duplicate kernel files with short names
- Kept descriptive long-named versions
- Cleaned up backup files

---

## What Needs To Be Done

### Phase 1: Implement v18 (NEXT STEP) 🎯
**Priority**: HIGHEST  
**Effort**: 5-7 hours  
**Risk**: Low  
**Expected**: 335-340 GFLOPS

**Tasks**:
1. [ ] Create `cuda_gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix.cu`
   - Copy from v5
   - Add stmatrix store logic from v12
   - Add padding (B_WG_M_PADDED = B_WG_M + 8)
   
2. [ ] Update `kebab/include/kebab/cuda/cuda_gemm.h`
   - Add v18 function declaration
   
3. [ ] Update `kebab/lib/CMakeLists.txt`
   - Add v18 to build list
   
4. [ ] Update `kebab/lib/cuda/cuda_gemm.cu`
   - Add case 18 to dispatch
   - Add feature name "wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix"
   
5. [ ] Update `config.yaml`
   - Add version 18 to test list
   
6. [ ] Build and test
   ```bash
   rm -rf kebab/build
   cmake -S kebab -B kebab/build
   cmake --build kebab/build -j$(nproc)
   ```
   
7. [ ] Verify correctness
   ```bash
   kebab/build/lib/benchmark/runonce_gemm_cuda 18 8192 RC random
   ```
   
8. [ ] Benchmark performance
   ```bash
   make bench-gemm
   ```
   
9. [ ] Analyze results
   - Compare vs v5 (332 GFLOPS)
   - Compare vs v15 (335 GFLOPS)
   - Check all matrix sizes (2048, 4096, 8192)

**Success Criteria**:
- ✅ Correctness test passes
- ✅ Performance >= 335 GFLOPS (match v15)
- ✅ Performance >= 337 GFLOPS (beat v15 by 0.6%)
- 🎯 Performance >= 340 GFLOPS (stretch goal)

**Reference**: See `V18_IMPLEMENTATION_PLAN.md` for detailed instructions

### Phase 2: Implement v19 (If v18 succeeds)
**Priority**: HIGH  
**Effort**: 6-8 hours  
**Risk**: Low  
**Expected**: 340-345 GFLOPS

**Tasks**:
1. [ ] Copy v5 kernel structure
2. [ ] Add Hilbert curve scheduling from v11
3. [ ] Test and benchmark

### Phase 3: Implement v21 (Ultimate kernel)
**Priority**: MEDIUM  
**Effort**: 10-12 hours  
**Risk**: Medium  
**Expected**: 345-355 GFLOPS

**Tasks**:
1. [ ] Start from v18 (v5 + stmatrix)
2. [ ] Add Hilbert scheduling
3. [ ] Test and benchmark
4. [ ] Select best kernel as default

### Phase 4: Optional - Implement v20
**Priority**: LOW  
**Effort**: 8-10 hours  
**Risk**: Medium  
**Expected**: 340-350 GFLOPS

**Tasks**:
1. [ ] Copy v15 kernel structure
2. [ ] Add warpgroup execution model
3. [ ] Test and benchmark

---

## Performance Targets

| Kernel | Status | Target GFLOPS | % of cuBLAS | vs v15 |
|--------|--------|---------------|-------------|--------|
| v15 (current best) | ✅ Done | 335 | 79.4% | baseline |
| v18 (v5 + stmatrix) | 🎯 Next | 335-340 | 79-81% | +0-5 |
| v19 (v5 + hilbert) | ⏳ Planned | 340-345 | 81-82% | +5-10 |
| v21 (v5 + stmx + hilb) | ⏳ Planned | 345-355 | 82-84% | +10-20 |
| v20 (v15 + warpgroup) | ⏳ Optional | 340-350 | 81-83% | +5-15 |

**Overall Goal**: Achieve 350+ GFLOPS (83% of cuBLAS)

---

## Timeline Estimate

- **Week 1**: Implement and test v18 (5-7 hours)
- **Week 2**: If successful, implement v19 (6-8 hours)
- **Week 3**: Implement v21 and final benchmarking (10-12 hours)
- **Total**: 21-27 hours over 3 weeks

---

## Key Files Reference

### Analysis Documents
- `analysis_gemm_features.md` - Feature taxonomy and performance data
- `GEMM_OPTIMIZATION_RECOMMENDATIONS.md` - Strategic recommendations
- `KERNEL_COMPARISON_TABLE.md` - Feature comparison table
- `GEMM_ANALYSIS_SUMMARY.md` - Executive summary

### Implementation Guides
- `V18_IMPLEMENTATION_PLAN.md` - Complete step-by-step guide for v18
- `DEVELOPMENT_STATUS.md` - This file (current status and next steps)

### Benchmark Results
- `bench_results/gemm_results_float16_cuda.csv` - Current performance data
- `bench_results/gemm_results_bfloat16_cuda.csv` - BF16 performance data

### Source Code
- `kebab/lib/cuda/cuda_gemm_v5_*.cu` - Base kernel for v18
- `kebab/lib/cuda/cuda_gemm_v12_*.cu` - Reference for stmatrix
- `kebab/lib/cuda/cuda_gemm_v11_*.cu` - Reference for Hilbert
- `kebab/lib/cuda/cuda_gemm_v15_*.cu` - Current best kernel

---

## Decision Points

### After v18 Implementation

**If v18 >= 337 GFLOPS** (beats v15):
- ✅ Proceed with v19 (v5 + hilbert)
- ✅ Proceed with v21 (v5 + stmatrix + hilbert)
- Goal: Achieve 350+ GFLOPS

**If v18 = 335-337 GFLOPS** (matches v15):
- ⚠️ Analyze why stmatrix didn't help more
- ⚠️ Consider v19 (v5 + hilbert) instead
- ⚠️ May need to tune padding or other parameters

**If v18 < 335 GFLOPS** (worse than v15):
- ❌ Debug stmatrix implementation
- ❌ Check for register spilling or bank conflicts
- ❌ Consider v19 (v5 + hilbert) as alternative

---

## Risk Mitigation

### Low Risk Items (v18, v19)
- Both use proven components
- Minimal architectural changes
- Easy to debug and tune
- High confidence in success

### Medium Risk Items (v20, v21)
- Multiple simultaneous changes
- Potential interaction effects
- May require more tuning
- Implement only if v18 succeeds

---

## Success Metrics

### Minimum Success
- v18 matches v15 (335 GFLOPS)
- No correctness regressions
- Stable across matrix sizes

### Target Success
- v18 beats v15 by 1% (338 GFLOPS)
- v19 achieves 340+ GFLOPS
- Clear path to 350+ GFLOPS

### Stretch Goal
- v21 achieves 350+ GFLOPS (83% of cuBLAS)
- Becomes new default kernel
- 5-6% improvement over current best

---

## Notes for Future Developers

### Why v5 is the Base
v5 achieves 332 GFLOPS with just 3 features:
- Warp group (128 threads)
- Warp specialization (producer/consumer)
- Persistent kernel (reuse thread blocks)

This simplicity is a strength, not a weakness. Adding only proven optimizations (stmatrix, hilbert) to this solid foundation is more likely to succeed than complex feature combinations.

### Why Not Start from v15?
v15 is current best (335 GFLOPS) but uses non-persistent scheduling. Adding warpgroup to v15 (v20) is riskier because:
- Architectural change (256 → 384 threads)
- Non-persistent + warpgroup interaction unknown
- More complex to implement

Starting from v5 is lower risk with similar upside.

### Key Insights
1. **Simplicity wins**: v5 beats 10 more complex kernels
2. **Clusters hurt**: -2% overhead, avoid them
3. **Hilbert helps**: +3.7% improvement
4. **stmatrix helps**: +0.6% improvement
5. **Persistent context-dependent**: Helps simple kernels, hurts complex ones

---

## Quick Start for Next Developer

1. **Read this file** to understand current status
2. **Read `V18_IMPLEMENTATION_PLAN.md`** for detailed instructions
3. **Start implementing v18** following the step-by-step guide
4. **Benchmark and compare** against v5 and v15
5. **Update this file** with results and next steps

---

## Contact & Questions

If you have questions about:
- **Analysis methodology**: See `analysis_gemm_features.md`
- **Strategic decisions**: See `GEMM_OPTIMIZATION_RECOMMENDATIONS.md`
- **Implementation details**: See `V18_IMPLEMENTATION_PLAN.md`
- **Feature comparisons**: See `KERNEL_COMPARISON_TABLE.md`

---

**Last Updated**: March 1, 2026  
**Status**: Ready for v18 implementation  
**Next Action**: Follow `V18_IMPLEMENTATION_PLAN.md`
