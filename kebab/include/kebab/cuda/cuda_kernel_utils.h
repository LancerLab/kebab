/**
 * @file cuda_kernel_utils.h
 * @brief Common utilities for CUDA kernels (WGMMA, TMA, etc.)
 *
 * This header provides shared utilities for all CUDA kernels:
 * - WGMMA descriptor generation with template parameters
 * - Swizzle pattern definitions
 * - Major order definitions
 * - WGMMA synchronization primitives
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <type_traits>

namespace baseline {
namespace cuda_kernel {

// ============================================================================
// WGMMA Enums and Constants
// ============================================================================

/**
 * @brief WGMMA swizzle pattern enum
 * Refer to: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
 * Section 9.7.15.5.1.2.2. Matrix Descriptor Format
 */
enum class WGMMA_Swizzle : uint64_t {
  NS = 0,    // No swizzle
  B32 = 3,   // 32B swizzle
  B64 = 2,   // 64B swizzle
  B128 = 1   // 128B swizzle
};

/**
 * @brief WGMMA major order enum
 */
enum class WGMMA_MajorOrder {
  K_MAJOR,   // K dimension is major (leading)
  MN_MAJOR   // M and N dimensions are major (leading)
};

/**
 * @brief WGMMA MMA shape enum
 * Defines the M, N, K dimensions as a single unit since they are interdependent
 */
enum class WGMMA_MMAShape {
  M64N64K16  // 64x64x16 matrix multiply (currently the only supported shape)
};

/**
 * @brief Extract M dimension from MMAShape
 */
template<WGMMA_MMAShape Shape>
constexpr int get_mma_m() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 64;
  return 0;
}

/**
 * @brief Extract N dimension from MMAShape
 */
template<WGMMA_MMAShape Shape>
constexpr int get_mma_n() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 64;
  return 0;
}

/**
 * @brief Extract K dimension from MMAShape
 */
template<WGMMA_MMAShape Shape>
constexpr int get_mma_k() {
  if constexpr (Shape == WGMMA_MMAShape::M64N64K16) return 16;
  return 0;
}

/**
 * @brief Infer TransA from MajorOrder
 * K_MAJOR = 0 (no transpose), MN_MAJOR = 1 (transpose)
 */
template<WGMMA_MajorOrder MajorOrder>
constexpr int get_trans_a() {
  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) return 0;
  if constexpr (MajorOrder == WGMMA_MajorOrder::MN_MAJOR) return 1;
  return 0;
}

/**
 * @brief Infer TransB from MajorOrder
 * K_MAJOR = 0 (no transpose), MN_MAJOR = 1 (transpose)
 */
template<WGMMA_MajorOrder MajorOrder>
constexpr int get_trans_b() {
  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) return 0;
  if constexpr (MajorOrder == WGMMA_MajorOrder::MN_MAJOR) return 1;
  return 0;
}

// ============================================================================
// Descriptor Encoding Utilities
// ============================================================================

/**
 * @brief Encode value for matrix descriptor
 * Extracts bits [17:4] from the input value
 */
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
}

// ============================================================================
// WGMMA Descriptor Creation (Templatized)
// ============================================================================

/**
 * @brief Create WGMMA shared memory descriptor with template parameters
 * 
 * @tparam MajorOrder K_MAJOR or MN_MAJOR layout
 * @tparam Swizzle Swizzle pattern (NS, B32, B64, B128)
 * @tparam T Data type (__half or __nv_bfloat16)
 * @param ptr Shared memory pointer
 * @return Encoded descriptor
 */
template <WGMMA_MajorOrder MajorOrder, WGMMA_Swizzle Swizzle, typename T>
__device__ static inline uint64_t make_smem_desc(T* ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0x0000000000000000;
  desc |= matrix_descriptor_encode(addr);

  // Determine stride and leading dimension based on major order and swizzle
  uint64_t stride_bytes = 0;
  uint64_t leading_dim = 0;

  if constexpr (MajorOrder == WGMMA_MajorOrder::K_MAJOR) {
    // K-major layout: stride varies by swizzle pattern
    switch (Swizzle) {
    case WGMMA_Swizzle::NS:
      stride_bytes = 128;
      leading_dim = 64;
      break;
    case WGMMA_Swizzle::B32:
      stride_bytes = 16;
      leading_dim = 256;
      break;
    case WGMMA_Swizzle::B64:
      stride_bytes = 16;
      leading_dim = 512;
      break;
    case WGMMA_Swizzle::B128:
      stride_bytes = 16;
      leading_dim = 1024;
      break;
    }
  }
  // TODO: MN-major not handled
  
  desc |= matrix_descriptor_encode(stride_bytes) << 16;
  desc |= matrix_descriptor_encode(leading_dim) << 32;
  desc |= static_cast<uint64_t>(Swizzle) << 62;

  return desc;
}

// ============================================================================
// WGMMA Synchronization Primitives
// ============================================================================

/**
 * @brief WGMMA fence synchronization
 */
__device__ static inline void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/**
 * @brief WGMMA commit batch
 */
__device__ static inline void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief WGMMA wait for N groups to complete
 * @tparam N Number of groups to wait for (0-7)
 */
template <int N>
__device__ static inline void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// ============================================================================
// WGMMA Core Instruction Wrappers
// ============================================================================

/**
 * @brief Generic WGMMA instruction wrapper (64x64x16)
 * Supports both FP16 and BF16 data types
 *
 * @tparam MajorOrderA Major order for matrix A (K_MAJOR or MN_MAJOR)
 * @tparam MajorOrderB Major order for matrix B (K_MAJOR or MN_MAJOR)
 * @tparam SwizzleA Swizzle pattern for matrix A
 * @tparam SwizzleB Swizzle pattern for matrix B
 * @tparam T Data type (__half or __nv_bfloat16)
 * @param d Output accumulator (4x8 floats)
 * @param sA Shared memory pointer to matrix A
 * @param sB Shared memory pointer to matrix B
 */
template<WGMMA_MajorOrder MajorOrderA, WGMMA_MajorOrder MajorOrderB,
         WGMMA_Swizzle SwizzleA, WGMMA_Swizzle SwizzleB, typename T>
__device__ static inline void wgmma_m64n64k16(float d[4][8], T* sA, T* sB) {
    // Infer Trans flags from MajorOrder (default scale values: 1 for identity scaling)
    constexpr int TransA = get_trans_a<MajorOrderA>();
    constexpr int TransB = get_trans_b<MajorOrderB>();
    constexpr int ScaleD = 1;  // Identity scale for accumulator
    constexpr int ScaleA = 1;  // Identity scale for matrix A
    constexpr int ScaleB = 1;  // Identity scale for matrix B

    uint64_t desc_a = make_smem_desc<MajorOrderA, SwizzleA, T>(&sA[0]);
    uint64_t desc_b = make_smem_desc<MajorOrderB, SwizzleB, T>(&sB[0]);

    if constexpr (std::is_same_v<T, __half>) {
        asm volatile(
            "{\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
            "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
            " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
            " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
            " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
            " %32,"
            " %33,"
            " %34, %35, %36, %37, %38;\n"
            "}\n"
            : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
            "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
            "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
            "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
            : "l"(desc_a), "l"(desc_b),
            "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        asm volatile(
            "{\n"
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
            "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
            " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
            " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
            " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
            " %32,"
            " %33,"
            " %34, %35, %36, %37, %38;\n"
            "}\n"
            : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]),
            "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]),
            "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
            "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]),
            "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]),
            "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
            "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]),
            "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
            : "l"(desc_a), "l"(desc_b),
            "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
    }
}

// ============================================================================
// Generic WGMMA Template (Parameterized by MMAShape and Data Type)
// ============================================================================

/**
 * @brief Generic WGMMA instruction using MMAShape enum
 *
 * @tparam Shape MMAShape enum (e.g., M64N64K16)
 * @tparam MajorOrderA Major order for matrix A
 * @tparam MajorOrderB Major order for matrix B
 * @tparam SwizzleA Swizzle pattern for matrix A
 * @tparam SwizzleB Swizzle pattern for matrix B
 * @tparam T Data type (__half or __nv_bfloat16)
 */
template<WGMMA_MMAShape Shape,
         WGMMA_MajorOrder MajorOrderA, WGMMA_MajorOrder MajorOrderB,
         WGMMA_Swizzle SwizzleA, WGMMA_Swizzle SwizzleB, typename T>
__device__ static inline void wgmma(float d[4][8], T* sA, T* sB) {
    static_assert(Shape == WGMMA_MMAShape::M64N64K16,
                  "wgmma: Currently only M64N64K16 is supported");
    wgmma_m64n64k16<MajorOrderA, MajorOrderB, SwizzleA, SwizzleB, T>(d, sA, sB);
}

// ============================================================================
// TensorMap Creation Utilities (Host-side)
// ============================================================================

/**
 * @brief Create TensorMap descriptor for TMA operations
 *
 * @tparam Swizzle Swizzle pattern (NS, B32, B64, B128)
 * @tparam BlockMajorSize Block size in major dimension
 * @tparam BlockMinorSize Block size in minor dimension
 * @tparam T Data type (__half or __nv_bfloat16)
 * @param tma_map Output TensorMap descriptor
 * @param gmem_ptr Global memory pointer
 * @param blocks_height Number of blocks in height dimension
 * @param blocks_width Number of blocks in width dimension
 */
template <WGMMA_Swizzle Swizzle, int BlockMajorSize, int BlockMinorSize, typename T>
inline void create_tensor_map_tma(CUtensorMap *tma_map, T* gmem_ptr,
                                   int blocks_height, int blocks_width) {
    void* gmem_address = (void*)gmem_ptr; // NOLINT
    uint64_t gmem_prob_shape[5] = {
        (uint64_t)BlockMinorSize * blocks_width,
        (uint64_t)BlockMajorSize * blocks_height,
        1, 1, 1
    };
    uint64_t gmem_prob_stride[5] = {
        sizeof(T),
        sizeof(T) * BlockMinorSize * blocks_width,
        0, 0, 0
    };
    uint32_t smem_box_shape[5] = {
        uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1
    };
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};
    CUresult result = CUDA_SUCCESS;
    CUtensorMapDataType tensormap_dtype;

    if constexpr (std::is_same_v<T, __half>) {
        tensormap_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        tensormap_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else {
        assert(false && "Unsupported data type for TMA");
    }

    if constexpr (Swizzle == WGMMA_Swizzle::NS) {
        result = cuTensorMapEncodeTiled(
            tma_map, tensormap_dtype, 2, gmem_address, gmem_prob_shape,
            gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    } else if constexpr (Swizzle == WGMMA_Swizzle::B128) {
        result = cuTensorMapEncodeTiled(
            tma_map, tensormap_dtype, 2, gmem_address, gmem_prob_shape,
            gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    } else if constexpr (Swizzle == WGMMA_Swizzle::B64) {
        result = cuTensorMapEncodeTiled(
            tma_map, tensormap_dtype, 2, gmem_address, gmem_prob_shape,
            gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    } else if constexpr (Swizzle == WGMMA_Swizzle::B32) {
        result = cuTensorMapEncodeTiled(
            tma_map, tensormap_dtype, 2, gmem_address, gmem_prob_shape,
            gmem_prob_stride + 1, smem_box_shape, smem_box_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "ERROR: cuTensorMapEncodeTiled failed with error %d\n", result);
    }
}

/**
 * @brief Allocate and create TensorMap descriptor
 *
 * @tparam Swizzle Swizzle pattern
 * @tparam BlockMajorSize Block size in major dimension
 * @tparam BlockMinorSize Block size in minor dimension
 * @tparam T Data type
 * @param gmem_ptr Global memory pointer
 * @param blocks_height Number of blocks in height dimension
 * @param blocks_width Number of blocks in width dimension
 * @return Pointer to device-side TensorMap descriptor
 */
template <WGMMA_Swizzle Swizzle, int BlockMajorSize, int BlockMinorSize, typename T>
inline CUtensorMap* allocate_and_create_tensor_map_tma(T* gmem_ptr,
                                                        int blocks_height, int blocks_width) {
    CUtensorMap *tma_map_d;
    cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
    CUtensorMap tma_map_host;
    create_tensor_map_tma<Swizzle, BlockMajorSize, BlockMinorSize, T>(
        &tma_map_host, gmem_ptr, blocks_height, blocks_width);
    cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    return tma_map_d;
}

/**
 * @brief Free TensorMap descriptor allocated on device
 */
inline void free_tensor_map_tma(CUtensorMap* tma_map_d) {
    if (tma_map_d) {
        cudaFree(tma_map_d);
    }
}

} // namespace cuda_kernel
} // namespace baseline

