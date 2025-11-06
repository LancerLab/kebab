/**
 * @file gemm_wgmma_tma.cu
 * @brief GEMM using Hopper WGMMA with TMA (Tensor Memory Accelerator)
 * 
 * Version 2: WGMMA with TMA for improved memory bandwidth
 * 
 * Based on: cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
 * 
 * TMA (Tensor Memory Accelerator) is a Hopper feature that:
 * - Offloads data movement from threads to dedicated hardware
 * - Supports multi-dimensional tensor addressing
 * - Provides better memory bandwidth utilization
 * - Reduces register pressure
 */

#include "kebab/cute/gemm.h"
#include "kebab/config/config_parser.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/device_kernel.h"

namespace kebab {
namespace cute {

using namespace ::cute;  // Use global cute namespace

using half_t = ::cute::half_t;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorageTMA
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device_tma(ProblemShape shape_MNK, CtaTiler cta_tiler,
                TA const* A_ptr, CUTLASS_GRID_CONSTANT TmaA const tma_a,
                TB const* B_ptr, CUTLASS_GRID_CONSTANT TmaB const tma_b,
                TC      * C_ptr, CStride dC, TiledMma mma,
                Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));

  // Full and Tiled Tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), make_shape(M,N), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorageTMA<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  // TMA partitioning
  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  // Pipeline setup
  auto K_PIPE_MAX = size<1>(tAsA);
  int k_tile_count = size<1>(tAgA);
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
  using ConsumerBarType = cutlass::arch::ClusterBarrier;
  
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  cluster_sync();

  // Prefetch
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  // MMA partitioning
  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  // Pipelined main loop
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();

  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
    warpgroup_commit_batch();

    warpgroup_wait<0>();

    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate)
    {
      int pipe = write_state.index();
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  // Epilogue
  axpby(alpha, tCrC, beta, tCgC);
}

// NT GEMM with TMA
template <class TA, class TB, class TC,
          class Alpha, class Beta,
          int BLK_M = 128, int BLK_N = 128, int BLK_K = 64>
void
gemm_nt_tma(int m, int n, int k,
            Alpha alpha,
            TA const* A, int ldA,
            TB const* B, int ldB,
            Beta beta,
            TC      * C, int ldC,
            cudaStream_t stream = 0)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(Int<1>{}, ldA);
  auto dB = make_stride(Int<1>{}, ldB);
  auto dC = make_stride(Int<1>{}, ldC);

  // Define CTA tile sizes (static) - now configurable via template parameters
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<  3>{};

  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  int smem_size = int(sizeof(SharedStorageTMA<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(4, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device_tma<decltype(prob_shape), decltype(cta_tiler),
                                               TA, decltype(sA), decltype(tmaA),
                                               TB, decltype(sB), decltype(tmaB),
                                               TC, decltype(dC), decltype(tiled_mma),
                                               decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();
}

// TN GEMM with TMA
template <class TA, class TB, class TC,
          class Alpha, class Beta,
          int BLK_M = 128, int BLK_N = 128, int BLK_K = 64>
void
gemm_tn_tma(int m, int n, int k,
            Alpha alpha,
            TA const* A, int ldA,
            TB const* B, int ldB,
            Beta beta,
            TC      * C, int ldC,
            cudaStream_t stream = 0)
{
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(ldA, Int<1>{});
  auto dB = make_stride(ldB, Int<1>{});
  auto dC = make_stride(Int<1>{}, ldC);

  // Define CTA tile sizes (static) - now configurable via template parameters
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = Int<3>{};

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  int smem_size = int(sizeof(SharedStorageTMA<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(4, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device_tma<decltype(prob_shape), decltype(cta_tiler),
                                               TA, decltype(sA), decltype(tmaA),
                                               TB, decltype(sB), decltype(tmaB),
                                               TC, decltype(dC), decltype(tiled_mma),
                                               decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();
}

/**
 * @brief WGMMA with TMA dispatch (Version 2) with configurable tile sizes
 */
void gemm_wgmma_tma_fp16_dispatch(const void* A_ptr, const void* B_ptr, void* C_ptr,
                                  int M, int N, int K, 
                                  char lhs_format, char rhs_format,
                                  int tile_M, int tile_N, int tile_K,
                                  cudaStream_t stream)
{
    const half_t* A = reinterpret_cast<const half_t*>(A_ptr);
    const half_t* B = reinterpret_cast<const half_t*>(B_ptr);
    half_t* C = reinterpret_cast<half_t*>(C_ptr);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    if (lhs_format == 'R' && rhs_format == 'C') {
        // RC mode: A row-major, B column-major -> use NT layout
        // Dispatch based on tile configuration
        if (tile_M == 64 && tile_N == 64 && tile_K == 32) {
            gemm_nt_tma<half_t, half_t, half_t, float, float, 64, 64, 32>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 128 && tile_K == 64) {
            gemm_nt_tma<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else if (tile_M == 256 && tile_N == 128 && tile_K == 64) {
            gemm_nt_tma<half_t, half_t, half_t, float, float, 256, 128, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 256 && tile_K == 64) {
            gemm_nt_tma<half_t, half_t, half_t, float, float, 128, 256, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else {
            // Default fallback
            fprintf(stderr, "Warning: Tile size [%d,%d,%d] not supported by TMA, using default [128,128,64]\n", 
                    tile_M, tile_N, tile_K);
            gemm_nt_tma<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        }
    } else if (lhs_format == 'C' && rhs_format == 'R') {
        // CR mode: A column-major, B row-major -> use TN layout
        // Dispatch based on tile configuration
        if (tile_M == 64 && tile_N == 64 && tile_K == 32) {
            // Test if 64x64x32 works with TMA TN layout
            fprintf(stderr, "Warning: Tile size [%d,%d,%d] may not be supported by TMA TN due to GMMA constraints, using default [128,128,64]\n", 
                    tile_M, tile_N, tile_K);
            gemm_tn_tma<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 128 && tile_K == 64) {
            gemm_tn_tma<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else if (tile_M == 256 && tile_N == 128 && tile_K == 64) {
            gemm_tn_tma<half_t, half_t, half_t, float, float, 256, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 256 && tile_K == 64) {
            gemm_tn_tma<half_t, half_t, half_t, float, float, 128, 256, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else {
            // Default fallback
            fprintf(stderr, "Warning: Tile size [%d,%d,%d] not supported by TMA TN, using default [128,128,64]\n", 
                    tile_M, tile_N, tile_K);
            gemm_tn_tma<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        }
    } else {
        fprintf(stderr, "ERROR: Invalid storage format combination for TMA: lhs=%c, rhs=%c\n", 
                lhs_format, rhs_format);
    }
}

/**
 * @brief WGMMA with TMA dispatch with config support
 */
void gemm_wgmma_tma_fp16(const void* A_ptr, const void* B_ptr, void* C_ptr,
                         int M, int N, int K, 
                         char lhs_format, char rhs_format,
                         cudaStream_t stream)
{
    try {
        // Get configuration instance
        auto& config = kebab::config::ConfigParser::getInstance();
        
        // Get tile sizes from configuration
        auto tile_sizes = config.getOperatorTileSizes("gemm");
        
        int tile_M = 128, tile_N = 128, tile_K = 64; // Default values
        
        if (tile_sizes.size() >= 3) {
            tile_M = tile_sizes[0];
            tile_N = tile_sizes[1];
            tile_K = tile_sizes[2];
        }
        
        // Call dispatch function with configured tile sizes
        gemm_wgmma_tma_fp16_dispatch(A_ptr, B_ptr, C_ptr, M, N, K, 
                                    lhs_format, rhs_format, 
                                    tile_M, tile_N, tile_K, stream);
    } catch (const std::exception& e) {
        // Fallback to default tile sizes if config loading fails
        fprintf(stderr, "Warning: Failed to load tile sizes from config for TMA: %s\n", e.what());
        fprintf(stderr, "Using default tile sizes: 128x128x64\n");
        gemm_wgmma_tma_fp16_dispatch(A_ptr, B_ptr, C_ptr, M, N, K, 
                                    lhs_format, rhs_format, 
                                    128, 128, 64, stream);
    }
}

} // namespace cute
} // namespace kebab
