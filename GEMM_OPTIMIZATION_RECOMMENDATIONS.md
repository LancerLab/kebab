# GEMM Kernel Optimization Recommendations

## Executive Summary

After analyzing all 17 GEMM kernel versions and their benchmark results (FP16, M=N=K=8192), I've identified promising feature combinations that are NOT currently implemented and likely to outperform the current best kernel (v15: 335 GFLOPS).

## Current Performance Landscape

### Top 3 Performers
1. **v15** (335,009 GFLOPS, 79.4% of cuBLAS) - warpspec + nopersistent + tmastore + hilbert + stmatrix + nocluster
2. **v5** (332,073 GFLOPS, 78.7% of cuBLAS) - warpgroup + warpspec + persistent (SIMPLE!)
3. **v14** (298,557 GFLOPS, 70.8% of cuBLAS) - warpspec + persistent + tmastore + hilbert + stmatrix + nocluster

### Key Insights

1. **Simplicity Wins**: v5 with minimal features (just warpgroup + warpspec + persistent) achieves 332 GFLOPS, beating many complex kernels
2. **Non-persistent is Better**: v15 (non-persistent) beats v14 (persistent) by 12% despite identical features
3. **Clusters Hurt**: v14 (no cluster) matches v12 (cluster) despite same features
4. **Hilbert Helps**: +3.7% improvement (v11 vs v10)
5. **stmatrix Helps**: +0.6% improvement (v12 vs v11)

## Missing High-Value Combinations

### Priority 1: v18 - v5 + stmatrix (HIGHEST CONFIDENCE)
**Expected**: 335-340 GFLOPS (beat current best)

**Rationale**:
- v5 is surprisingly good (332 GFLOPS) with minimal complexity
- stmatrix adds +0.6% in v12 vs v11
- No cluster overhead (clusters hurt performance)
- Simple persistent scheduling works well

**Implementation**:
- Copy v5 kernel structure
- Add stmatrix store logic from v12 (lines 450-460)
- Add padding to shared memory (B_WG_M_PADDED = B_WG_M + 8)
- Keep v5's simple persistent scheduling

**Estimated effort**: 4-6 hours

### Priority 2: v19 - v5 + hilbert (HIGH CONFIDENCE)
**Expected**: 340-345 GFLOPS

**Rationale**:
- v5 base is proven (332 GFLOPS)
- Hilbert adds +3.7% (v11 vs v10)
- Combined: 332 * 1.037 = 344 GFLOPS

**Implementation**:
- Copy v5 kernel structure
- Add Hilbert curve tile scheduling from v11
- Keep v5's warpgroup + persistent model

**Estimated effort**: 6-8 hours

### Priority 3: v20 - v15 + warpgroup (MEDIUM CONFIDENCE)
**Expected**: 340-350 GFLOPS

**Rationale**:
- v15 is current best (335 GFLOPS)
- Test if warpgroup helps non-persistent kernels
- v5 shows warpgroup is valuable (+8% vs v4)

**Implementation**:
- Copy v15 kernel structure
- Change from warpspec-only to warpgroup model
- Increase thread count from 256 to 384 (3 warp-groups)

**Estimated effort**: 8-10 hours

### Priority 4: v21 - v5 + stmatrix + hilbert (ULTIMATE)
**Expected**: 345-355 GFLOPS

**Rationale**:
- Combine all proven optimizations on v5 base
- stmatrix: +0.6%, hilbert: +3.7%
- Combined: 332 * 1.006 * 1.037 = 346 GFLOPS

**Implementation**:
- Start from v18 (v5 + stmatrix)
- Add Hilbert scheduling from v11
- Most complex but highest potential

**Estimated effort**: 10-12 hours

## Feature Impact Analysis

| Feature | Impact | Evidence | Recommendation |
|---------|--------|----------|----------------|
| Warp specialization | +8% | v4 vs v3 | ✅ Always use |
| Persistent (simple) | +37% | v5 vs v4 | ✅ Use for warpgroup |
| Non-persistent | +12% | v15 vs v14 | ✅ Test both modes |
| Hilbert scheduling | +3.7% | v11 vs v10 | ✅ Add to v5 |
| stmatrix | +0.6% | v12 vs v11 | ✅ Add to v5 |
| Warp group | +8% | v3 vs v2 | ✅ Use with persistent |
| Cluster | -2% | v8 vs v7 | ❌ Avoid |
| TMA store | -1.5% | v10 vs v8 | ⚠️ Marginal |
| Stream store | -3% | v9 vs v8 | ❌ Avoid |

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. Implement v18 (v5 + stmatrix)
2. Benchmark against v5 and v15
3. If successful, proceed to Phase 2

### Phase 2: Advanced Optimizations (2-3 days)
1. Implement v19 (v5 + hilbert)
2. Implement v20 (v15 + warpgroup)
3. Benchmark all variants

### Phase 3: Ultimate Kernel (1-2 days)
1. Implement v21 (v5 + stmatrix + hilbert)
2. Final benchmarking and tuning
3. Select best kernel as default

## Expected Outcomes

**Conservative estimate**: 340 GFLOPS (81% of cuBLAS) with v18
**Optimistic estimate**: 355 GFLOPS (84% of cuBLAS) with v21
**Current best**: 335 GFLOPS (79% of cuBLAS) with v15

**Improvement**: +5-20 GFLOPS (+1.5-6%)

## Technical Details for Implementation

### v18 Implementation Guide

**Base**: `cuda_gemm_v5_wgmma_tma_warpgroup_warpspecialized_persistent.cu`

**Changes needed**:
1. Add padding constant:
   ```cuda
   constexpr int V18_B_WG_M_PADDED = V18_B_WG_M + 8;  // 64 + 8 = 72
   ```

2. Update shared memory structure:
   ```cuda
   __half C_smem[NUM_CONSUMERS][B_WG_M_PADDED * BN];  // Was B_WG_M * BN
   ```

3. Replace float-to-half conversion + store with stmatrix:
   ```cuda
   // OLD (v5):
   for (int m_it = 0; m_it < M_ITERATIONS; ++m_it) {
       for (int w = 0; w < 256; w += 16) {
           C_smem[consumer_idx][w * B_WG_M + m_it * WGMMA_M + lane_id] = 
               __float2half(d[m_it][w / 16][lane_id % 8]);
       }
   }
   
   // NEW (v18):
   __half d_fp16[8];
   int* data_ptr = (int*)d_fp16;
   for (int m_it = 0; m_it < M_ITERATIONS; ++m_it) {
       for (int w = 0; w < 256; w += 16) {
           uint32_t addr = base_addr + (w * B_WG_M_PADDED + m_it * WGMMA_M) * sizeof(__half);
           for (int k = 0; k < 8; k++) d_fp16[k] = __float2half(d[m_it][w / 16][k]);
           asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], {%1, %2, %3, %4};"
                       :: "r"(addr), "r"(data_ptr[0]), "r"(data_ptr[1]), 
                          "r"(data_ptr[2]), "r"(data_ptr[3]));
       }
   }
   ```

4. Update TMA tensor map for C output:
   ```cuda
   // Use B_WG_M_PADDED instead of B_WG_M for stride calculation
   ```

### Files to Modify

1. **kebab/lib/cuda/cuda_gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix.cu** (NEW)
2. **kebab/include/kebab/cuda/cuda_gemm.h** - Add function declaration
3. **kebab/lib/cuda/CMakeLists.txt** - Add v18 to build
4. **kebab/lib/cuda/cuda_gemm.cu** - Add case 18 dispatch + feature name

## Benchmark Configuration

Test with:
- Matrix sizes: 2048, 4096, 8192, 16384
- Precision: FP16, BF16
- Mode: RC (row-major A, column-major B)
- Compare against: v5, v15, cuBLAS

## Success Criteria

- v18 beats v15 (335 GFLOPS) by at least 1%
- v18 beats v5 (332 GFLOPS) by at least 2%
- No correctness regressions
- Stable performance across matrix sizes

## Risk Assessment

**Low Risk**:
- v18 (v5 + stmatrix): Proven components, minimal changes
- v19 (v5 + hilbert): Both features proven separately

**Medium Risk**:
- v20 (v15 + warpgroup): Architectural change to non-persistent kernel
- v21 (v5 + stmatrix + hilbert): Multiple changes, interaction effects

## Conclusion

The analysis reveals that **v5's simplicity is a strength**, not a weakness. By adding only proven optimizations (stmatrix, hilbert) to v5's solid foundation, we can likely achieve 340-355 GFLOPS, beating the current best by 5-20 GFLOPS.

**Recommended action**: Implement v18 first as a proof of concept. If successful (>337 GFLOPS), proceed with v19 and v21.
