# Task Quick Reference

## Current Status
✅ **Phase 1 Complete**: Foundation + Basic Implementations
- Framework, benchmarking, profiling ✅
- Element-wise add (CuTe) ✅
- GEMM basic (clean, correct) ✅
- Performance: 12,327 GFLOPS (54% of cuBLAS)

## Next: Phase 2 - GEMM Optimization (CRITICAL)

### Phase 2A: WGMMA Tensor Cores (6 tasks)
🎯 **Target**: 70-80% of cuBLAS  
⏱️ **Time**: 1-2 weeks  
📈 **Expected**: 10-20x speedup for FP16

**Tasks**:
1. [ ] Study CUTLASS WGMMA examples (2-3 hours)
2. [ ] Implement WGMMA atom for FP16
3. [ ] Implement swizzled shared memory layouts
4. [ ] Implement WGMMA compute loop
5. [ ] Optimize tile sizes (128x128, 128x256, 256x128)
6. [ ] Benchmark and validate

**Start Here**: `cutlass/examples/cute/tutorial/wgmma_gemm.cu`

### Phase 2B: TMA Async Copy (6 tasks)
🎯 **Target**: 80-90% of cuBLAS  
⏱️ **Time**: 1-2 weeks  
📈 **Expected**: 1.5-2x improvement

**Tasks**:
1. [ ] Study CUTLASS TMA examples
2. [ ] Create TMA descriptors
3. [ ] Implement TMA async loads
4. [ ] Implement 3-stage pipeline
5. [ ] Optimize TMA parameters
6. [ ] Benchmark and validate

### Phase 2C: Thread Block Clusters (5 tasks)
🎯 **Target**: 90-95% of cuBLAS  
⏱️ **Time**: 1 week  
📈 **Expected**: 1.2-1.3x improvement

### Phase 2D: FP8 & Final (4 tasks)
🎯 **Target**: 95%+ of cuBLAS  
⏱️ **Time**: 1 week  
📈 **Expected**: 2x throughput with FP8

**Total Phase 2**: 4-6 weeks

## After Phase 2: Other Operators

### Phase 3: Convolution (13 tasks, 3-6 weeks)
- 3A: Basic (50-60% of cuDNN)
- 3B: Winograd (70-80% of cuDNN)
- 3C: Optimized (80-90% of cuDNN)

### Phase 4: Reduction (7 tasks, 2 weeks)
- 4A: Warp-level (60-70% of CUB)
- 4B: Multi-stage (80-90% of CUB)

### Phase 5: Additional Operators (6 tasks, 2-3 weeks)
- Softmax
- LayerNorm
- Batched GEMM
- Grouped GEMM

### Phase 6: Final Integration (5 tasks, 1-2 weeks)
- Performance validation
- Documentation
- Testing

## Total Timeline
- **Minimum**: 12-15 weeks
- **Expected**: 15-20 weeks
- **With buffer**: 20-24 weeks (5-6 months)

## Key Principles

1. **Pure CuTe**: No raw CUDA in `src/`
2. **Hardware Features**: Use Tensor Cores, TMA, clusters
3. **Verify Everything**: Run actual benchmarks
4. **Performance Target**: ≥90% of vendor libraries
5. **No Placeholders**: Real implementations only

## Documentation

- **Detailed Tasks**: `.kiro/specs/cute-kernel-bench/tasks.md`
- **Roadmap**: `docs/TASK_ROADMAP.md`
- **Current Performance**: `docs/PERFORMANCE_BASELINE.md`
- **Implementation Guide**: `docs/CUTE_GEMM_IMPLEMENTATION.md`
- **Status Tracking**: `docs/IMPLEMENTATION_STATUS.md`

## Commands

```bash
# Build
make clean && make build

# Benchmark
make bench-gemm
make bench-elementwise-add

# Profile
make tune-gemm

# Verify
./scripts/verify_cute_implementation.sh

# Check status
cat docs/PERFORMANCE_BASELINE.md
cat FINAL_STATUS.md
```

## Success Criteria

### Performance Targets
- ✅ Element-wise: ~96% of CUDA baseline
- 🎯 GEMM: ≥95% of cuBLAS (Phase 2 goal)
- 🎯 Conv2D: ≥90% of cuDNN (Phase 3 goal)
- 🎯 Reduction: ≥85% of CUB (Phase 4 goal)

### Code Quality
- ✅ Pure CuTe implementations
- ✅ All tests passing
- ✅ Complete documentation
- 🎯 All hardware features utilized

## Next Action

**Start Phase 2A Task 1**: Study CUTLASS WGMMA examples
- Read: `cutlass/examples/cute/tutorial/wgmma_gemm.cu`
- Read: `cutlass/examples/cute/tutorial/sgemm_sm90.cu`
- Document key concepts
- Time: 2-3 hours

**Then**: Implement WGMMA atom (Task 2)
