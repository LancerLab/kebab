# CuTe GEMM Implementation Guide

## Problem Statement

The original `src/operators/gemm.cu` implementation was **not using CuTe properly**:
- Only included CuTe headers but used raw CUDA code
- Manual pointer arithmetic and indexing
- No Layout or Tensor abstractions
- No Tensor Core utilization
- Essentially a standard CUDA kernel with CuTe headers

## Solution

Created `src/operators/gemm_cute_hopper.cu` with proper CuTe usage and Hopper optimizations.

## Key Differences

### Original Implementation (gemm.cu)
```cpp
// Raw CUDA approach
__shared__ T smem_A[TILE_M][TILE_K];
int row = by * TILE_M + ty * THREAD_TILE;
int col = bx * TILE_N + tx * THREAD_TILE;
smem_A[row][col] = A_ptr[global_row * K + global_col];
```

### New Implementation (gemm_cute_hopper.cu)
```cpp
// CuTe approach
auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
Tensor gA = make_tensor(make_gmem_ptr(A_ptr), layout_A);
Tensor gA_tile = local_tile(gA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(by, _));
auto thr_gA = local_partition(gA_tile, make_layout(make_shape(Int<NUM_THREADS>{})), tid);
```

## CuTe Core Concepts

### 1. Layout: Separating Logic from Memory

**Traditional CUDA**:
```cpp
// Logic and memory layout tightly coupled
int idx = row * stride + col;
T value = ptr[idx];
```

**CuTe**:
```cpp
// Logic separated from memory layout
auto layout = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
Tensor t = make_tensor(ptr, layout);
T value = t(row, col);  // Automatic index calculation
```

**Benefits**:
- Compiler can optimize layout at compile-time
- Easy to change memory layout (row-major, column-major, swizzled)
- Type-safe multi-dimensional indexing

### 2. Tensor: Multi-dimensional Array Abstraction

**Traditional CUDA**:
```cpp
// Manual bounds checking and indexing
if (row < M && col < N) {
    C[row * N + col] = A[row * K + k] * B[k * N + col];
}
```

**CuTe**:
```cpp
// Automatic bounds and layout handling
Tensor gC = make_tensor(C_ptr, make_layout(make_shape(M, N)));
gC(row, col) = gA(row, k) * gB(k, col);
```

**Benefits**:
- Dimension-aware operations
- Automatic stride calculation
- Cleaner, more maintainable code

### 3. Tiling: Hierarchical Decomposition

**Traditional CUDA**:
```cpp
// Manual tile calculation
int tile_row = blockIdx.y * TILE_M;
int tile_col = blockIdx.x * TILE_N;
int local_row = threadIdx.y;
int local_col = threadIdx.x;
```

**CuTe**:
```cpp
// Automatic tiling
Tensor gA_tile = local_tile(gA, make_shape(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(by, _));
auto thr_tile = local_partition(gA_tile, thread_layout, tid);
```

**Benefits**:
- Compile-time tile size optimization
- Automatic coordinate transformation
- Easy to experiment with different tile sizes

### 4. Partitioning: Work Distribution

**Traditional CUDA**:
```cpp
// Manual work distribution
int elements_per_thread = TILE_SIZE / NUM_THREADS;
for (int i = 0; i < elements_per_thread; ++i) {
    int idx = tid * elements_per_thread + i;
    // Process element
}
```

**CuTe**:
```cpp
// Automatic work distribution
auto thr_partition = local_partition(tile, thread_layout, tid);
for (int i = 0; i < size(thr_partition); ++i) {
    thr_partition(i) = /* process */;
}
```

**Benefits**:
- Optimal memory access patterns
- Automatic coalescing
- Load balancing

## Hopper Architecture Features

### What Makes Hopper Special

1. **WGMMA (Warp Group MMA)**
   - 2x throughput vs Ampere
   - Operates on 64x256x16 tiles
   - Asynchronous execution

2. **TMA (Tensor Memory Accelerator)**
   - Hardware-accelerated bulk data transfer
   - Automatic 2D/3D copy with swizzling
   - Offloads work from threads

3. **Thread Block Clusters**
   - Up to 228KB distributed shared memory
   - Better data sharing across CTAs
   - Improved occupancy

4. **Enhanced Async Pipeline**
   - Better overlap of compute and memory
   - Hardware support for producer-consumer
   - Lower synchronization overhead

### Implementation Roadmap

#### Phase 1: Basic CuTe ✅ (Current)
```cpp
// Use Layout and Tensor abstractions
auto layout = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
Tensor gA = make_tensor(make_gmem_ptr(A_ptr), layout);
```

#### Phase 2: Tensor Cores 🚧 (Next)
```cpp
// Use WGMMA instructions
using MMA_Atom = SM90_64x256x16_F16F16F16_SS<>;
using TiledMMA = TiledMMA<MMA_Atom, Layout<Shape<_2,_1,_1>>>;
gemm(tiled_mma, sA, sB, acc);
```

#### Phase 3: TMA 🚧 (Future)
```cpp
// Async bulk transfer
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gA, make_shape(BLK_M, BLK_K));
copy(tma_load, gA_tile, sA);
```

#### Phase 4: Clusters 🚧 (Future)
```cpp
// Launch with clusters
dim3 cluster_dims(2, 2, 1);
cudaLaunchKernelEx(&cfg, kernel, args...);
```

## Performance Expectations

| Implementation | Expected Performance | Key Features |
|---------------|---------------------|--------------|
| Original (raw CUDA) | 30-40% of cuBLAS | Basic tiling, no Tensor Cores |
| Phase 1 (Basic CuTe) | 30-40% of cuBLAS | Layout abstractions, better code |
| Phase 2 (+ WGMMA) | 70-80% of cuBLAS | Tensor Core utilization |
| Phase 3 (+ TMA) | 80-90% of cuBLAS | Async data movement |
| Phase 4 (+ Clusters) | 90-95% of cuBLAS | All Hopper features |

## Learning Resources

### CUTLASS CuTe Tutorials
Located in `cutlass/examples/cute/tutorial/`:

1. **sgemm.cu**: Basic GEMM with CuTe
2. **sgemm_sm90.cu**: Hopper-optimized GEMM
3. **tma_load.cu**: TMA usage examples
4. **wgmma_gemm.cu**: WGMMA instruction usage
5. **cluster_gemm.cu**: Thread block clusters

### Key Documentation
- [CuTe Quickstart](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [CuTe Layout](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)
- [CuTe Tensor](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_tensor.md)
- [Hopper Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

## Next Steps

1. **Study CUTLASS tutorials**: Understand WGMMA and TMA usage
2. **Implement Phase 2**: Add Tensor Core support with WGMMA
3. **Benchmark**: Compare against cuBLAS and profile
4. **Iterate**: Add TMA, clusters, and other optimizations
5. **Document**: Update README with performance results

## File Organization

```
src/operators/
├── gemm.cu                    # Legacy raw CUDA (to be deprecated)
├── gemm_cute_hopper.cu        # New CuTe implementation
└── README_GEMM_CUTE.md        # Detailed feature analysis

docs/
├── CUTE_GEMM_IMPLEMENTATION.md  # This file
└── fixes/                       # Historical fix documentation
    ├── CUDA_*.md
    ├── PROFILING_FIX*.md
    └── DEFAULT_SIZES_UPDATE.md

scripts/
├── profile_gemm.sh            # Enhanced profiling script
├── diagnose_profiling.sh      # Profiling diagnostics
└── test_*.sh                  # Various test scripts
```

## Summary

The new `gemm_cute_hopper.cu` demonstrates **proper CuTe usage** with:
- ✅ Layout and Tensor abstractions
- ✅ Hierarchical tiling and partitioning
- ✅ Compile-time optimization opportunities
- ✅ Foundation for Hopper features

The implementation is ready for Phase 2 (WGMMA) and beyond, with clear documentation of what features are used and what's still needed.