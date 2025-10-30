# GEMM Performance Bottleneck Analysis & Further Optimization Plan

## Current Performance Status
- **Current**: 12303 GFLOPS (53.9% of cuBLAS)
- **Target**: 80-90% of cuBLAS (~18000-20000 GFLOPS)

## Critical Bottlenecks Identified from NCU Profiling

### 1. **Shared Memory Bank Conflicts** (Est. Speedup: 29.61%) 🔴 CRITICAL
```
- 3.0-way bank conflicts on shared stores
- 32,768 bank conflicts out of 49,152 wavefronts (66.67%)
- Caused by: smem_A[row][col] and smem_B[row][col] access patterns
```

**Root Cause**: 
- Current layout: `smem_A[64][16]` and `smem_B[16][64]`
- When threads in a warp access consecutive columns, they hit the same bank
- 16 columns = 16 banks, but with stride causing conflicts

**Solution**: Pad shared memory to avoid bank conflicts
```cuda
__shared__ T smem_A[TILE_M][TILE_K + 8];  // Add padding
__shared__ T smem_B[TILE_K][TILE_N + 8];  // Add padding
```

### 2. **L1TEX Memory Stalls** (Est. Speedup: 44.57%) 🔴 CRITICAL
```
- 3.4 cycles per warp stalled on L1TEX operations
- Represents 44.6% of total stall time
- L1/TEX Hit Rate: 59.40% (should be >80%)
```

**Root Cause**:
- Sequential loading without prefetching
- No overlap between compute and memory operations
- Single buffering in shared memory

**Solution**: Software pipelining with double buffering
- Use `cp.async` for asynchronous memory loads
- Overlap next tile load with current tile compute
- Requires double-buffered shared memory

### 3. **Poor Memory Coalescing** (Est. Speedup: 3.36% + 4.20%)
```
- Global loads: Only 12.8/32 bytes utilized per sector (40% efficiency)
- Global stores: Only 8.0/32 bytes utilized per sector (25% efficiency)
```

**Root Cause**:
- Small tile loading (only 16 threads load 16 elements)
- Inefficient access patterns in cooperative loading

**Solution**: Vectorized loads using float4
```cuda
float4* A_vec = (float4*)A_ptr;
// Load 4 floats at once per thread
```

### 4. **Low Scheduler Utilization** (Est. Speedup: 73.11%) 🔴 CRITICAL
```
- Only 26.89% of cycles have eligible warps
- 73.11% of cycles have no eligible warps
- Average 2.04 active warps per scheduler (max 16)
- Only 0.39 eligible warps per cycle
```

**Root Cause**:
- Low occupancy (12.47% achieved)
- Too few warps to hide latency
- Small block size (256 threads = 8 warps)

**Solution**: Increase block size and tile size
- Use 32x8 threads (256 threads, but better warp distribution)
- Or increase to 512 threads per block (16 warps)
- Larger tiles: 128x128 instead of 64x64

### 5. **No Tensor Core Utilization** 🟡 MAJOR OPPORTUNITY
```
- Current: Using FP32 FMA units
- Potential: 8-16x speedup with Tensor Cores
```

**Solution**: Use CuTe MMA atoms or WMMA
- For H100: Use `SM90_16x8x16_F32F16F16F32_TN` MMA atom
- Requires FP16 input, FP32 accumulation
- Can achieve 300+ TFLOPS on H100

## Optimization Priority & Implementation Plan

### Phase 1: Quick Wins (Est. 35-40% speedup)
1. **Fix Shared Memory Bank Conflicts** (29.61% speedup)
   - Add padding to shared memory arrays
   - Change: `smem_A[64][16]` → `smem_A[64][24]`
   - Effort: 5 minutes

2. **Vectorized Memory Access** (7.56% speedup)
   - Use float4 for loading A and B tiles
   - Coalesce 4 floats per thread
   - Effort: 15 minutes

### Phase 2: Software Pipelining (Est. 40-50% speedup)
3. **Double Buffering + cp.async** (44.57% speedup)
   - Implement double-buffered shared memory
   - Use async copy to overlap compute and memory
   - Requires: 2x shared memory, pipeline logic
   - Effort: 1-2 hours

### Phase 3: Occupancy Optimization (Est. 20-30% speedup)
4. **Increase Block Size** (partial 73.11% speedup)
   - Change from 16x16 to 32x8 or 16x32 threads
   - Increase tile size to 128x128
   - Better warp utilization
   - Effort: 30 minutes

### Phase 4: Tensor Core Integration (Est. 8-16x speedup)
5. **Use CuTe MMA Atoms**
   - Implement Tensor Core GEMM with CuTe
   - Use SM90 MMA atoms for H100
   - Requires: FP16 data path, TiledMMA
   - Effort: 4-8 hours

## Expected Performance After Each Phase

| Phase | Optimization | Est. GFLOPS | % of cuBLAS | Cumulative Speedup |
|-------|--------------|-------------|-------------|-------------------|
| Current | Baseline | 12,303 | 53.9% | 1.0x |
| Phase 1 | Bank Conflict + Vectorize | 17,000 | 74.5% | 1.38x |
| Phase 2 | + Software Pipeline | 22,000 | 96.3% | 1.79x |
| Phase 3 | + Occupancy | 24,000 | 105% | 1.95x |
| Phase 4 | + Tensor Cores | 150,000+ | 650%+ | 12x+ |

## Recommended Next Steps

**Immediate (Phase 1)**: Fix bank conflicts and add vectorization
- Low effort, high impact
- Should reach ~75% of cuBLAS

**Short-term (Phase 2)**: Implement software pipelining
- Moderate effort, very high impact
- Should reach ~95% of cuBLAS

**Long-term (Phase 4)**: Tensor Core integration
- High effort, transformative impact
- Should exceed cuBLAS for large matrices

## Code Changes Required

### 1. Bank Conflict Fix (Immediate)
```cuda
// Before:
__shared__ T smem_A[TILE_M][TILE_K];
__shared__ T smem_B[TILE_K][TILE_N];

// After:
__shared__ T smem_A[TILE_M][TILE_K + 8];  // Padding to avoid conflicts
__shared__ T smem_B[TILE_K][TILE_N + 8];
```

### 2. Vectorized Loads (Immediate)
```cuda
// Load 4 floats at once
if (global_col % 4 == 0 && global_col + 3 < K) {
    float4 val = *((float4*)&A_ptr[global_row * K + global_col]);
    smem_A[row][col] = val.x;
    smem_A[row][col+1] = val.y;
    smem_A[row][col+2] = val.z;
    smem_A[row][col+3] = val.w;
}
```

### 3. Double Buffering (Short-term)
```cuda
__shared__ T smem_A[2][TILE_M][TILE_K + 8];  // Double buffer
__shared__ T smem_B[2][TILE_K][TILE_N + 8];

int write_idx = 0;
int read_idx = 1;

// Pipeline: load next while computing current
```

## Conclusion

The current implementation has achieved 53.9% of cuBLAS through basic tiling. The profiling data reveals three critical bottlenecks:

1. **Shared memory bank conflicts** (29.61% potential speedup)
2. **L1TEX memory stalls** (44.57% potential speedup)  
3. **Low scheduler utilization** (73.11% potential speedup)

By addressing these in phases, we can realistically achieve:
- **Phase 1**: ~75% of cuBLAS (quick wins)
- **Phase 2**: ~95% of cuBLAS (with pipelining)
- **Phase 4**: Exceed cuBLAS (with Tensor Cores)

The next immediate action should be implementing Phase 1 optimizations (bank conflict fix + vectorization) as they provide significant gains with minimal effort.
