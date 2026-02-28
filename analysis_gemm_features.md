# GEMM Kernel Feature Analysis

## Performance Summary (FP16, M=N=K=8192)

| Version | GFLOPS | Speedup | Key Features |
|---------|--------|---------|--------------|
| v15 | 335,009 | 0.794 | **BEST** - warpspec + nopersistent + tmastore + hilbert + stmatrix + nocluster |
| v5 | 332,073 | 0.787 | warpgroup + warpspec + persistent |
| v14 | 298,557 | 0.708 | warpspec + persistent + tmastore + hilbert + stmatrix + nocluster |
| v12 | 299,458 | 0.710 | warpgroup + warpspec + persistent + cluster + tmastore + hilbert + stmatrix |
| v11 | 297,562 | 0.705 | warpgroup + warpspec + persistent + cluster + tmastore + hilbert |
| v16 | 294,717 | 0.699 | warpspec + persistent + tmastore + hilbert + stmatrix + nocluster + linear |
| v8 | 292,903 | 0.694 | warpgroup + warpspec + persistent + cluster + ptxbarrier + tma5d |
| v13 | 287,097 | 0.681 | warpspec + persistent + ptxbarrier + tma2d |
| v10 | 287,040 | 0.680 | warpgroup + warpspec + persistent + cluster + tmastore |
| v7 | 286,049 | 0.678 | warpgroup + warpspec + persistent + ptxbarrier + tma5d |
| v9 | 283,830 | 0.673 | warpgroup + warpspec + persistent + cluster + streamstore |
| v4 | 242,317 | 0.574 | warpgroup + warpspec |
| v6 | 250,582 | 0.594 | warpgroup + warpspec + persistent + tilescheduler |
| v3 | 224,858 | 0.533 | warpgroup |
| v2 | 169,168 | 0.401 | wgmma + tma |

## Feature Taxonomy

### Core Features
1. **WGMMA** - Warp Group Matrix Multiply-Accumulate (all versions)
2. **TMA** - Tensor Memory Accelerator (all versions)

### Execution Model
3. **Warp Group** - 128 threads (4 warps) working together
4. **Warp Specialization** - Producer/consumer warp roles
5. **Persistent Kernel** - Reuse thread blocks across tiles
6. **Non-Persistent** - Traditional one-tile-per-block

### Scheduling
7. **Tile Scheduler** - Dynamic work distribution
8. **Hilbert Curve** - Space-filling curve tile ordering
9. **Linear Schedule** - Sequential tile ordering

### Synchronization
10. **PTX Barrier** - Low-level barrier primitives
11. **TMA 5D** - 5-dimensional TMA descriptors
12. **TMA 2D** - 2-dimensional TMA descriptors

### Memory Optimization
13. **TMA Store** - Async TMA for stores
14. **Stream Store** - Streaming stores to global memory
15. **stmatrix** - Shared memory matrix store instruction
16. **Padded SMEM** - Padding to avoid bank conflicts

### Multi-SM
17. **Cluster** - Thread block clusters
18. **Multicast** - TMA multicast within cluster
19. **No Cluster** - Single thread block

## Key Insights

### Top Performers
- **v15** (335 GFLOPS): Non-persistent + stmatrix + no cluster
- **v5** (332 GFLOPS): Simple persistent warpspec (no advanced features!)

### Surprising Results
1. **Persistent hurts sometimes**: v15 (non-persistent) beats v14 (persistent) by 12%
2. **Clusters hurt**: v14 (no cluster) beats v12 (cluster) despite same features
3. **Simple wins**: v5 with minimal features beats complex v12
4. **Hilbert helps**: v11 (hilbert) beats v10 (no hilbert) by 3.7%
5. **stmatrix helps**: v12 (stmatrix) beats v11 (no stmatrix) by 0.6%

## Missing Promising Combinations

Based on analysis, these combinations are NOT implemented:

### High Priority (likely to beat v15's 335 GFLOPS)
1. **v5 + stmatrix + hilbert** - Take best simple kernel (v5) and add proven optimizations
2. **v15 + warpgroup** - Add warpgroup to best kernel
3. **v5 + tmastore + hilbert** - v5 base with async stores and better scheduling

### Medium Priority
4. **v7 + nocluster + stmatrix** - Remove cluster overhead, add stmatrix
5. **v13 + hilbert + stmatrix** - Add proven optimizations to tma2d variant
6. **v4 + tmastore + hilbert** - Non-persistent warpspec with async stores

### Experimental
7. **v5 + streamstore** - Test if streaming helps simple persistent
8. **v15 + tma2d** - Test if 2D TMA is better for non-persistent
9. **v3 + ptxbarrier + stmatrix** - Minimal warpgroup with modern sync

## Feature Impact Analysis

| Feature | Impact | Evidence |
|---------|--------|----------|
| Non-persistent | +12% | v15 vs v14 |
| No cluster | +0% | v14 vs v12 (similar) |
| Hilbert | +3.7% | v11 vs v10 |
| stmatrix | +0.6% | v12 vs v11 |
| Warp specialization | +8% | v4 vs v3 |
| Persistent (simple) | +37% | v5 vs v4 |
| TMA store | -1.5% | v10 vs v8 |
| Cluster | -2% | v8 vs v7 |

## Recommended Implementation Order

1. **v18**: v5 + stmatrix + hilbert (highest confidence)
2. **v19**: v15 + warpgroup (test if warpgroup helps non-persistent)
3. **v20**: v5 + tmastore + hilbert (test async stores on v5)
