/**
 * @file mbench_gmma_descriptor_info.cu
 * @brief Microbenchmark to print GMMA descriptor information for various configurations
 *
 * This microbench prints GMMA descriptor details (LBO, SBO, base_offset, layout_type)
 * for different layout atoms on SM90 Hopper architecture.
 *
 * IMPORTANT: This microbench uses simple 64x64 layout atoms. For actual GEMM kernels,
 * descriptors are generated from tensors created with tile_to_shape(Layout_XXX_Atom<T>{},
 * make_shape(M, K, P)), which produces different LBO/SBO values.
 *
 * Actual descriptor values from CUTE kernel (RR mode):
 * - K-major (A matrix): LBO=1, SBO=64
 * - MN-major (B matrix): LBO=0, SBO=128
 *
 * These values are used in cuda_gemm_v2_wgmma_tma.cu for manual descriptor generation.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <type_traits>

// Include CUTE headers for layout definitions
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

using namespace cute;

// ============================================================================
// Helper Functions to Extract Descriptor Fields
// ============================================================================

__device__ void print_descriptor_fields(const char* name, uint64_t desc) {
    uint16_t start_address = desc & 0x3FFF;
    uint16_t leading_byte_offset = (desc >> 16) & 0x3FFF;
    uint16_t stride_byte_offset = (desc >> 32) & 0x3FFF;
    uint8_t base_offset = (desc >> 49) & 0x7;
    uint8_t layout_type = (desc >> 62) & 0x3;

    printf("%s:\n", name);
    printf("  Raw: 0x%016llx\n", (unsigned long long)desc);
    printf("  start_addr: 0x%04x (%5d)\n", start_address, start_address);
    printf("  LBO:        0x%04x (%5d)\n", leading_byte_offset, leading_byte_offset);
    printf("  SBO:        0x%04x (%5d)\n", stride_byte_offset, stride_byte_offset);
    printf("  base_offset: %d\n", base_offset);
    printf("  layout_type: %d (%s)\n", layout_type,
           layout_type == 0 ? "INTERLEAVE" :
           layout_type == 1 ? "B128" :
           layout_type == 2 ? "B64" : "B32");
}

// ============================================================================
// Kernel: Print Descriptor Info for All Configurations
// ============================================================================

__global__ void kernel_print_gmma_descriptors() {
    if (threadIdx.x != 0) return;

    // Allocate shared memory for testing
    __shared__ alignas(128) __half smem_fp16[64 * 64];

    printf("\n");
    printf("=============================================================================\n");
    printf("GMMA Descriptor Information for SM90 Hopper\n");
    printf("=============================================================================\n");

    // ========================================================================
    // FP16 Configurations - Simple atoms (64x64)
    // ========================================================================
    printf("\n--- FP16 (half_t) Simple Atom Configurations (64x64) ---\n");

    // MN-major SW128
    {
        using MN_Layout = GMMA::Layout_MN_SW128_Atom<__half>;
        auto layout = MN_Layout{};
        auto tensor = make_tensor(make_smem_ptr(smem_fp16), layout);
        auto desc = SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(tensor);
        printf("\nFP16 MN-major SW128 (64x64):\n");
        print_descriptor_fields("  Descriptor", desc.desc_);
    }

    // MN-major SW64
    {
        using MN_Layout = GMMA::Layout_MN_SW64_Atom<__half>;
        auto layout = MN_Layout{};
        auto tensor = make_tensor(make_smem_ptr(smem_fp16), layout);
        auto desc = SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(tensor);
        printf("\nFP16 MN-major SW64 (64x64):\n");
        print_descriptor_fields("  Descriptor", desc.desc_);
    }

    // MN-major SW32
    {
        using MN_Layout = GMMA::Layout_MN_SW32_Atom<__half>;
        auto layout = MN_Layout{};
        auto tensor = make_tensor(make_smem_ptr(smem_fp16), layout);
        auto desc = SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(tensor);
        printf("\nFP16 MN-major SW32 (64x64):\n");
        print_descriptor_fields("  Descriptor", desc.desc_);
    }

    // MN-major INTERLEAVE
    {
        using MN_Layout = GMMA::Layout_MN_INTER_Atom<__half>;
        auto layout = MN_Layout{};
        auto tensor = make_tensor(make_smem_ptr(smem_fp16), layout);
        auto desc = SM90::GMMA::make_gmma_desc<SM90::GMMA::Major::MN>(tensor);
        printf("\nFP16 MN-major INTERLEAVE (64x64):\n");
        print_descriptor_fields("  Descriptor", desc.desc_);
    }

    // ========================================================================
    // Note on K-major Layouts
    // ========================================================================
    printf("\n--- K-major Layouts (Note) ---\n");
    printf("K-major atoms have K-size of 8, 16, 32, or 64 (in uint128_t units)\n");
    printf("But make_gmma_desc<Major::K> requires K-size of 2 or 4\n");
    printf("Therefore, K-major atoms cannot be directly used with make_gmma_desc\n");
    printf("K-major descriptors are generated from tile_to_shape results in actual kernels\n");
    printf("\n--- How to Extract Descriptors from tile_to_shape Results ---\n");
    printf("In actual GEMM kernels:\n");
    printf("  1. Create rank-3 layout: auto sA = tile_to_shape(Layout_K_SW128_Atom<T>{}, make_shape(M,K,P))\n");
    printf("  2. Extract rank-2 for TMA: auto tma_layout = sA(_,_,0)  // First pipeline stage\n");
    printf("  3. Extract rank-2 for MMA: auto mma_tensor = tensor<0>(make_tensor(ptr, sA))\n");
    printf("  4. Generate descriptor: auto desc = make_gmma_desc<Major::K>(mma_tensor)\n");
    printf("\nThe descriptor values depend on the specific M, K, P values used in tile_to_shape\n");

}

// ============================================================================
// Main Entry Point
// ============================================================================

int main() {
    printf("Launching GMMA Descriptor Info Kernel...\n");
    kernel_print_gmma_descriptors<<<1, 128>>>();
    cudaDeviceSynchronize();
    printf("Done!\n");
    return 0;
}

