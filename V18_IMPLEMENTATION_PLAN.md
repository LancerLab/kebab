# V18 Kernel Implementation Plan

## Overview
Implement v18 kernel: v5 (simple persistent warpgroup) + stmatrix optimization

**Expected performance**: 335-340 GFLOPS (beat current best v15: 335 GFLOPS)

## Step-by-Step Implementation

### Step 1: Create v18 Kernel File (2-3 hours)

**File**: `kebab/lib/cuda/cuda_gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix.cu`

**Actions**:
1. Copy `cuda_gemm_v5_wgmma_tma_warpgroup_warpspecialized_persistent.cu` to new file
2. Rename all v5 symbols to v18:
   - `V5_*` constants → `V18_*`
   - `_v5` function suffixes → `_v18`
   - `SMemV5` → `SMemV18`
   - etc.

3. Add padding constant after line ~200:
   ```cuda
   constexpr int V18_B_WG_M_PADDED = V18_B_WG_M + 8;  // 64 + 8 = 72
   ```

4. Update shared memory structure (around line ~220):
   ```cuda
   template <int BM, int BN, int BK, int QSIZE>
   struct SMemV18 {
       __half A[QSIZE][BM * BK];
       __half B[QSIZE][BK * BN];
       __half C[NUM_CONSUMERS][V18_B_WG_M_PADDED * BN];  // Changed from B_WG_M
       uint64_t barriers[QSIZE];
   };
   ```

5. Replace store logic in consumer warp section (around line ~400-420):
   
   **Find this code**:
   ```cuda
   // Store results to shared memory
   for (int m_it = 0; m_it < M_ITERATIONS; ++m_it) {
       for (int w = 0; w < 256; w += 16) {
           int smem_idx = w * B_WG_M + m_it * WGMMA_M + lane_id;
           smem.C[consumer_idx][smem_idx] = __float2half(d[m_it][w / 16][lane_id % 8]);
       }
   }
   ```
   
   **Replace with**:
   ```cuda
   // Store results to shared memory using stmatrix
   __half d_fp16[8];
   int* data_ptr = (int*)d_fp16;
   uint32_t base_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.C[consumer_idx][0]));
   
   for (int m_it = 0; m_it < M_ITERATIONS; ++m_it) {
       for (int w = 0; w < 256; w += 16) {
           // Convert float to half
           for (int k = 0; k < 8; k++) {
               d_fp16[k] = __float2half(d[m_it][w / 16][k]);
           }
           
           // Use stmatrix for efficient store
           uint32_t addr = base_addr + (w * V18_B_WG_M_PADDED + m_it * WGMMA_M) * sizeof(__half);
           asm volatile(
               "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(addr), "r"(data_ptr[0]), "r"(data_ptr[1]), 
                  "r"(data_ptr[2]), "r"(data_ptr[3])
           );
       }
   }
   ```

6. Update TMA tensor map creation for C (around line ~180):
   ```cuda
   template <int BlockMajorSize, int BlockMinorSize>
   void create_tensor_map_v18(CUtensorMap *tma_map, __half* gmem_ptr,
                               int blocks_height, int blocks_width) {
       // ... existing code ...
       
       // Update box shape to use padded size
       uint32_t smem_box_shape[5] = {
           BlockMinorSize,
           V18_B_WG_M_PADDED,  // Changed from BlockMajorSize
           1, 1, 1
       };
       
       // ... rest of function ...
   }
   ```

7. Update function signature and export (end of file):
   ```cuda
   void gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix_fp16(
       const __half* A, const __half* B, __half* C,
       int M, int N, int K, char lhs_format, char rhs_format,
       cudaStream_t stream) {
       // ... implementation ...
   }
   ```

### Step 2: Update Header File (15 minutes)

**File**: `kebab/include/kebab/cuda/cuda_gemm.h`

**Actions**:
1. Add declaration after v17 (around line ~197):
   ```cuda
   /**
    * @brief V18: v5 + stmatrix optimization
    * Simple persistent warpgroup kernel with efficient shared memory stores
    */
   void gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix_fp16(
       const __half* A, const __half* B, __half* C,
       int M, int N, int K, char lhs_format, char rhs_format,
       cudaStream_t stream);
   ```

### Step 3: Update CMakeLists.txt (5 minutes)

**File**: `kebab/lib/CMakeLists.txt`

**Actions**:
1. Add v18 to source list (after line ~28):
   ```cmake
   add_library(kebab_cuda
       # ... existing files ...
       cuda/cuda_gemm_v17_wgmma_tma_warpgroup_ptxbarrier.cu
       cuda/cuda_gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix.cu
   )
   ```

### Step 4: Update Dispatch Logic (15 minutes)

**File**: `kebab/lib/cuda/cuda_gemm.cu`

**Actions**:
1. Add feature name (around line ~40):
   ```cpp
   const char* gemm_cuda_version_feature_name(int version) {
       switch (version) {
           // ... existing cases ...
           case 17: return "wgmma_tma_warpgroup_ptxbarrier";
           case 18: return "wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix";
           default: return "unknown";
       }
   }
   ```

2. Add dispatch case (around line ~140):
   ```cpp
   void gemm(const __half* A, const __half* B, __half* C,
             int M, int N, int K, const char* opmode, int version, cudaStream_t stream) {
       // ... existing code ...
       
       switch (version) {
           // ... existing cases ...
           case 17:
               gemm_v17_wgmma_tma_warpgroup_ptxbarrier_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
               break;
           case 18:
               // V18: v5 + stmatrix (SM90 Hopper, RC mode only)
               gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix_fp16(A, B, C, M, N, K, lhs_format, rhs_format, stream);
               break;
           default:
               fprintf(stderr, "ERROR: Unsupported CUDA version %d\n", version);
               fprintf(stderr, "       Available: 1-18\n");
               return;
       }
   }
   ```

### Step 5: Update Config File (5 minutes)

**File**: `config.yaml`

**Actions**:
1. Add v18 to version list:
   ```yaml
   operators:
     gemm:
       impls: ["cuda"]
       versions: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
       # ... rest of config ...
   ```

### Step 6: Build (30 minutes)

**Commands**:
```bash
# Clean build
rm -rf kebab/build
cmake -S kebab -B kebab/build
cmake --build kebab/build -j$(nproc)
```

**Expected output**:
- No compilation errors
- New kernel compiled successfully
- All existing kernels still compile

### Step 7: Test Correctness (15 minutes)

**Commands**:
```bash
# Run single test
kebab/build/lib/benchmark/runonce_gemm_cuda 18 8192 RC random

# Check output
# Should see: "✓ Verification passed"
```

### Step 8: Benchmark (30 minutes)

**Commands**:
```bash
# Run full benchmark
make bench-gemm

# Or directly
kebab/build/lib/benchmark/bench_gemm
```

**Expected results**:
- v18 should show 335-340 GFLOPS at M=N=K=8192
- Should beat or match v15 (335 GFLOPS)
- Should beat v5 (332 GFLOPS) by 1-2%

### Step 9: Analysis (30 minutes)

**Actions**:
1. Check CSV results:
   ```bash
   cat bench_results/gemm_results_float16_cuda.csv | grep "cuda_v18"
   ```

2. Compare against v5 and v15:
   ```bash
   cat bench_results/gemm_results_float16_cuda.csv | grep -E "cuda_v(5|15|18)"
   ```

3. Create performance summary:
   - Plot GFLOPS vs matrix size
   - Calculate speedup vs v5
   - Calculate speedup vs v15
   - Identify any regressions

## Validation Checklist

- [ ] Code compiles without errors
- [ ] Correctness test passes (matches cuBLAS within tolerance)
- [ ] Performance >= v5 (332 GFLOPS)
- [ ] Performance >= v15 (335 GFLOPS) OR within 1%
- [ ] No memory errors (run with cuda-memcheck if available)
- [ ] Works for all matrix sizes (2048, 4096, 8192)

## Troubleshooting

### Compilation Errors

**Issue**: stmatrix instruction not recognized
**Solution**: Ensure `-arch=sm_90` flag is set in CMakeLists.txt

**Issue**: Undefined reference to v18 function
**Solution**: Check that function name matches exactly in .cu and .h files

### Correctness Errors

**Issue**: Results don't match cuBLAS
**Solution**: 
1. Check padding is applied consistently
2. Verify stmatrix address calculation
3. Test with smaller matrix first (M=N=K=256)

### Performance Issues

**Issue**: v18 slower than v5
**Solution**:
1. Check that stmatrix is actually being used (inspect PTX/SASS)
2. Verify padding doesn't cause bank conflicts
3. Try different padding values (4, 8, 16)

**Issue**: v18 much slower than expected
**Solution**:
1. Check for register spilling (use `--ptxas-options=-v`)
2. Verify shared memory size doesn't exceed limits
3. Profile with nsys/ncu to identify bottlenecks

## Success Metrics

**Minimum success**: v18 >= 335 GFLOPS (match v15)
**Target success**: v18 >= 337 GFLOPS (beat v15 by 0.6%)
**Stretch goal**: v18 >= 340 GFLOPS (beat v15 by 1.5%)

## Next Steps After v18

If v18 succeeds:
1. Implement v19 (v5 + hilbert) - Expected: 340-345 GFLOPS
2. Implement v21 (v5 + stmatrix + hilbert) - Expected: 345-355 GFLOPS
3. Select best kernel as default

If v18 fails:
1. Analyze why stmatrix didn't help
2. Try v19 (v5 + hilbert) instead
3. Consider v20 (v15 + warpgroup)

## Time Estimate

- Implementation: 3-4 hours
- Testing & debugging: 1-2 hours
- Benchmarking & analysis: 1 hour
- **Total**: 5-7 hours

## Files Modified Summary

1. **NEW**: `kebab/lib/cuda/cuda_gemm_v18_wgmma_tma_warpgroup_warpspecialized_persistent_stmatrix.cu`
2. **MODIFIED**: `kebab/include/kebab/cuda/cuda_gemm.h`
3. **MODIFIED**: `kebab/lib/CMakeLists.txt`
4. **MODIFIED**: `kebab/lib/cuda/cuda_gemm.cu`
5. **MODIFIED**: `config.yaml`
