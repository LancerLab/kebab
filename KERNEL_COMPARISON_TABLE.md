# GEMM Kernel Feature Comparison Table

## All Implemented Kernels (v1-v17)

| Ver | GFLOPS | %cuBLAS | WG | WS | Per | Tile | PTX | TMA | Clus | MC | Store | Hilb | stmx | Notes |
|-----|--------|---------|----|----|-----|------|-----|-----|------|----|----|------|------|-------|
| v1  | - | - | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | reg | ❌ | ❌ | Baseline warp tiling |
| v2  | 169 | 40% | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | reg | ❌ | ❌ | WGMMA + TMA |
| v3  | 225 | 53% | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | reg | ❌ | ❌ | +Warp group |
| v4  | 242 | 57% | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | reg | ❌ | ❌ | +Warp spec |
| v5  | 332 | 79% | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | reg | ❌ | ❌ | **+Persistent** ⭐ |
| v6  | 251 | 59% | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | reg | ❌ | ❌ | +Tile sched |
| v7  | 286 | 68% | ✅ | ✅ | ✅ | ✅ | ✅ | 5D | ❌ | ❌ | reg | ❌ | ❌ | +PTX barrier |
| v8  | 293 | 69% | ✅ | ✅ | ✅ | ✅ | ✅ | 5D | ✅ | ✅ | reg | ❌ | ❌ | +Cluster+MC |
| v9  | 284 | 67% | ✅ | ✅ | ✅ | ✅ | ✅ | 5D | ✅ | ✅ | strm | ❌ | ❌ | +Stream store ❌ |
| v10 | 287 | 68% | ✅ | ✅ | ✅ | ✅ | ✅ | 5D | ✅ | ✅ | tma | ❌ | ❌ | +TMA store |
| v11 | 298 | 71% | ✅ | ✅ | ✅ | ✅ | ✅ | 5D | ✅ | ✅ | tma | ✅ | ❌ | +Hilbert |
| v12 | 299 | 71% | ✅ | ✅ | ✅ | ✅ | ✅ | 5D | ✅ | ✅ | tma | ✅ | ✅ | +stmatrix |
| v13 | 287 | 68% | ❌ | ✅ | ✅ | ✅ | ✅ | 2D | ❌ | ❌ | reg | ❌ | ❌ | TMA 2D variant |
| v14 | 299 | 71% | ❌ | ✅ | ✅ | ✅ | ✅ | 5D | ❌ | ❌ | tma | ✅ | ✅ | -Cluster |
| v15 | 335 | 79% | ❌ | ✅ | ❌ | ✅ | ✅ | 5D | ❌ | ❌ | tma | ✅ | ✅ | **-Persistent** ⭐ |
| v16 | 295 | 70% | ❌ | ✅ | ✅ | lin | ✅ | 5D | ❌ | ❌ | tma | ✅ | ✅ | Linear sched |
| v17 | - | - | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | reg | ❌ | ❌ | PTX isolation test |

## Proposed New Kernels (v18-v21)

| Ver | Expected | %cuBLAS | WG | WS | Per | Tile | PTX | TMA | Clus | MC | Store | Hilb | stmx | Base | Changes |
|-----|----------|---------|----|----|-----|------|-----|-----|------|----|----|------|------|------|---------|
| v18 | 335-340 | 80-81% | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | reg | ❌ | ✅ | v5 | +stmatrix |
| v19 | 340-345 | 81-82% | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | reg | ✅ | ❌ | v5 | +hilbert |
| v20 | 340-350 | 81-83% | ✅ | ✅ | ❌ | ✅ | ✅ | 5D | ❌ | ❌ | tma | ✅ | ✅ | v15 | +warpgroup |
| v21 | 345-355 | 82-84% | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | reg | ✅ | ✅ | v5 | +stmx+hilb |

## Legend

**Execution Model**:
- **WG**: Warp Group (128 threads, 4 warps)
- **WS**: Warp Specialization (producer/consumer roles)
- **Per**: Persistent kernel (reuse thread blocks)

**Scheduling**:
- **Tile**: Tile scheduler (dynamic work distribution)
  - ✅ = dynamic scheduler
  - lin = linear static mapping
  - ❌ = no scheduler

**Synchronization**:
- **PTX**: PTX mbarrier primitives
- **TMA**: Tensor Memory Accelerator
  - 2D = 2-dimensional descriptors
  - 5D = 5-dimensional descriptors
  - ✅ = basic TMA

**Multi-SM**:
- **Clus**: Thread block clusters
- **MC**: TMA multicast within cluster

**Memory**:
- **Store**: Output store method
  - reg = register-based stores
  - strm = streaming stores (__stwt)
  - tma = TMA async stores
- **Hilb**: Hilbert curve tile ordering
- **stmx**: stmatrix instruction for shared memory

## Performance Patterns

### What Works
1. **Warp specialization**: +8% (v4 vs v3)
2. **Persistent (simple)**: +37% (v5 vs v4)
3. **Non-persistent (complex)**: +12% (v15 vs v14)
4. **Hilbert**: +3.7% (v11 vs v10)
5. **stmatrix**: +0.6% (v12 vs v11)

### What Doesn't Work
1. **Clusters**: -2% (v8 vs v7)
2. **Stream stores**: -3% (v9 vs v8)
3. **TMA stores**: -1.5% (v10 vs v8)
4. **Tile scheduler on persistent**: -24% (v6 vs v5)

### Key Insights

**Simple Persistent (v5)**:
- 332 GFLOPS with minimal features
- Beats 10 more complex kernels
- **Lesson**: Simplicity + proven features > complexity

**Non-Persistent (v15)**:
- 335 GFLOPS, current best
- Same features as v14 but -persistent
- **Lesson**: Persistent helps simple kernels, hurts complex ones

**Clusters Hurt**:
- v14 (no cluster) matches v12 (cluster)
- v8 (cluster) slower than v7 (no cluster)
- **Lesson**: Cluster overhead > multicast benefit

## Recommended Combinations

### Priority 1: v18 (v5 + stmatrix)
**Why**: Proven base + proven optimization
**Risk**: Low
**Effort**: 5-7 hours
**Expected**: 335-340 GFLOPS

### Priority 2: v19 (v5 + hilbert)
**Why**: Proven base + proven optimization
**Risk**: Low
**Effort**: 6-8 hours
**Expected**: 340-345 GFLOPS

### Priority 3: v21 (v5 + stmatrix + hilbert)
**Why**: Best of both optimizations
**Risk**: Medium (interaction effects)
**Effort**: 10-12 hours
**Expected**: 345-355 GFLOPS

### Priority 4: v20 (v15 + warpgroup)
**Why**: Test if warpgroup helps non-persistent
**Risk**: Medium (architectural change)
**Effort**: 8-10 hours
**Expected**: 340-350 GFLOPS

## Feature Combinations NOT Tested

These combinations exist in the feature space but are not implemented:

1. ✅ **v5 + stmatrix** (v18) - PROPOSED
2. ✅ **v5 + hilbert** (v19) - PROPOSED
3. ✅ **v5 + stmatrix + hilbert** (v21) - PROPOSED
4. ✅ **v15 + warpgroup** (v20) - PROPOSED
5. **v5 + TMA store** - Lower priority
6. **v5 + stream store** - Not recommended (v9 shows it hurts)
7. **v4 + stmatrix** - Non-persistent warpspec variant
8. **v4 + hilbert** - Non-persistent warpspec variant
9. **v3 + stmatrix** - Simple warpgroup variant
10. **v13 + stmatrix + hilbert** - TMA 2D variant

## Matrix Size Sensitivity

Performance at different sizes (FP16):

| Version | 2048 | 4096 | 8192 | Trend |
|---------|------|------|------|-------|
| v5 | 266 | 395 | 332 | Peak at 4K |
| v15 | 320 | 422 | 335 | Peak at 4K |
| v12 | 309 | 319 | 299 | Stable |

**Observation**: Most kernels peak at 4096, then drop at 8192
**Implication**: Test new kernels at multiple sizes

## Conclusion

The feature space analysis reveals:
1. **v5 is underoptimized** - simple base with room for improvement
2. **v15 is well-optimized** - but warpgroup might help
3. **4 promising gaps** in the feature space
4. **Expected gain**: 5-20 GFLOPS with proposed kernels

**Next action**: Implement v18 (v5 + stmatrix) as proof of concept.
