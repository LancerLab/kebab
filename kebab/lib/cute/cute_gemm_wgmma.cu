/**
 * @file gemm_wgmma.cu
 * @brief GEMM using Hopper WGMMA - Direct port from CUTLASS tutorial
 * 
 * Phase 2A Task 2: WGMMA Tensor Cores implementation
 * 
 * This is a direct port of cutlass/examples/cute/tutorial/hopper/wgmma_sm90.cu
 * Reference: cutlass/examples/cute/tutorial/hopper/wgmma_sm90.cu
 */

#include "kebab/cute/gemm.h"
#include "kebab/config/config_parser.h"

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include <string>

using namespace cute;

namespace kebab {
namespace cute {

using namespace ::cute;  // Use global cute namespace

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_);                           // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor sB_ = as_position_independent_swizzle_tensor(sB);
  Tensor tBsB = thr_copy_b.partition_D(sB_);                           // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCsA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCsB)));              // MMA_N
  CUTE_STATIC_ASSERT_V((size<2>(tCsA) == size<2>(tCsB)));              // MMA_K

  // Clear the accumulators
  clear(tCrC);

  // Total number of k-tiles
  auto K_TILE_MAX  = size<3>(tAgA);
  // Number of pipelined k-tiles in smem
  auto K_PIPE_MAX  = size<3>(tAsA);

  //
  // PREFETCH
  //

  // Prefetch all but the last
  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX-1; ++k)
  {
    copy(copy_a, tAgA(_,_,_,k), tAsA(_,_,_,k));
    copy(copy_b, tBgB(_,_,_,k), tBsB(_,_,_,k));
    cp_async_fence();
  }

  __syncthreads();

  //
  // PIPELINED MAIN LOOP
  //

  // Current pipe to read from
  int k_pipe_read  = 0;
  // Current pipe to write to
  int k_pipe_write = K_PIPE_MAX-1;

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    int k_tile_next = k_tile + (K_PIPE_MAX-1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX-1 : k_tile_next;

    //
    // Copy gmem to smem for k_tile_write
    //

    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe_write));
    cp_async_fence();

    // Advance k_pipe_write
    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    //
    // Compute on k_tile
    //

    // Wait on all cp.async
    cp_async_wait<0>();

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    gemm(mma, tCrA(_,_,_,k_pipe_read), tCrB(_,_,_,k_pipe_read), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // Advance k_pipe_read
    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta,
          int BLK_M = 128, int BLK_N = 128, int BLK_K = 64>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static) - now configurable via template parameters
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 32x4 m-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 32x4 n-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1 n-major

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  int  smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

  auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                  TA, decltype(dA), decltype(sA), decltype(copyA),
                                  TB, decltype(dB), decltype(sB), decltype(copyB),
                                  TC, decltype(dC), decltype(tiled_mma),
                                  decltype(alpha), decltype(beta)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, dA, sA, copyA,
                                                             B, dB, sB, copyB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "ERROR: WGMMA NT kernel launch failed\n");
  }
}

// Setup params for a NN GEMM (A: row-major, B: row-major)
template <class TA, class TB, class TC,
          class Alpha, class Beta,
          int BLK_M = 128, int BLK_N = 128, int BLK_K = 64>
void
gemm_nn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NN strides (both row-major)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK) - A row-major: A[m,k] = A[m*K + k]
  auto dB = make_stride(ldB, Int<1>{});                      // (dK, dN) - B row-major: B[k,n] = B[k*N + n]
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static) - now configurable via template parameters
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});              // Val layout  1x8
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 16x8 n-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::MN>{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  int  smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

  auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                  TA, decltype(dA), decltype(sA), decltype(copyA),
                                  TB, decltype(dB), decltype(sB), decltype(copyB),
                                  TC, decltype(dC), decltype(tiled_mma),
                                  decltype(alpha), decltype(beta)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, dA, sA, copyA,
                                                             B, dB, sB, copyB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "ERROR: WGMMA NN kernel launch failed\n");
  }
}

// Setup params for a TT GEMM (A^T: col-major, B^T: col-major)
template <class TA, class TB, class TC,
          class Alpha, class Beta,
          int BLK_M = 128, int BLK_N = 128, int BLK_K = 64>
void
gemm_tt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TT strides (both col-major)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK) - A^T col-major
  auto dB = make_stride(Int<1>{}, ldB);                      // (dK, dN) - B^T col-major
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static) - now configurable via template parameters
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 16x8 m-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 16x8 n-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(4, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  int  smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

  auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                  TA, decltype(dA), decltype(sA), decltype(copyA),
                                  TB, decltype(dB), decltype(sB), decltype(copyB),
                                  TC, decltype(dC), decltype(tiled_mma),
                                  decltype(alpha), decltype(beta)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, dA, sA, copyA,
                                                             B, dB, sB, copyB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "ERROR: WGMMA TT kernel launch failed\n");
  }
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta,
          int BLK_M = 128, int BLK_N = 128, int BLK_K = 64>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static) - now configurable via template parameters
  auto bM = Int<BLK_M>{};
  auto bN = Int<BLK_N>{};
  auto bK = Int<BLK_K>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});              // Val layout  1x8
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{}, // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});              // Val layout  1x8

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::K,GMMA::Major::K>{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(4, 2, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                           TA, decltype(dA), decltype(sA), decltype(copyA),
                                           TB, decltype(dB), decltype(sB), decltype(copyB),
                                           TC, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, dA, sA, copyA,
                                                             B, dB, sB, copyB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "ERROR: WGMMA TN kernel launch failed\n");
  }
}

/**
 * @brief Wrapper for FP16 GEMM using WGMMA
 *
 * Input: Row-major matrices A (M×K), B (K×N)
 * Output: Row-major matrix C (M×N)
 *
 * Row-major layout means:
 * - A[i,j] is at A[i*K + j]
 * - B[i,j] is at B[i*N + j]
 * - C[i,j] is at C[i*N + j]
 *
 * Uses TN layout from reference code with row-major strides.
 */
/**
 * @brief Complete WGMMA-based GEMM dispatch for FP16 (Hopper SM90+)
 * 
 * @param A_ptr Input matrix A (M×K, row-major)
 * @param B_ptr Input matrix B (K×N, row-major)  
 * @param C_ptr Output matrix C (M×N, row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param lhs_mode "N" for normal A, "T" for transpose A
 * @param rhs_mode "N" for normal B, "T" for transpose B
 * @param tile_M Tile size for M dimension
 * @param tile_N Tile size for N dimension
 * @param tile_K Tile size for K dimension
 * @param stream CUDA stream
 */
void gemm_wgmma_fp16_dispatch(const void* A_ptr, const void* B_ptr, void* C_ptr,
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
    
    // CORRECTED IMPLEMENTATION: Align with cuBLAS naming conventions
    // cuBLAS: opA refers to operation on A, opB refers to operation on B
    // CuTe: gemm_tn means A^T × B, gemm_nt means A × B^T
    // 
    // cuBLAS call: cublasSgemm(handle, opB, opA, N, M, K, alpha, B, ldB, A, ldA, beta, C, ldC)
    // This computes: C^T = opB(B) × opA(A)
    //
    // To align CuTe with cuBLAS naming:
    // - When cuBLAS uses opA=T, opB=N: C^T = B × A^T, we should call gemm_nt (A^T × B after swap)
    // - When cuBLAS uses opA=N, opB=T: C^T = B^T × A, we should call gemm_tn (A × B^T after swap)
    
    // Dispatch based on tile sizes and storage format
    if (lhs_format == 'R' && rhs_format == 'C') {
        // RC: cuBLAS uses opA=T, opB=N => C^T = B × A^T
        // CuTe: gemm_nt computes C^T = A^T × B with row-major A,B
        // After argument swap: gemm_nt(N, M, K, B, ldB, A, ldA, C, ldC)
        
        // Dispatch based on tile configuration
        if (tile_M == 64 && tile_N == 64 && tile_K == 32) {
            gemm_nt<half_t, half_t, half_t, float, float, 64, 64, 32>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 128 && tile_K == 64) {
            gemm_nt<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else if (tile_M == 256 && tile_N == 128 && tile_K == 64) {
            gemm_nt<half_t, half_t, half_t, float, float, 256, 128, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 256 && tile_K == 64) {
            gemm_nt<half_t, half_t, half_t, float, float, 128, 256, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        } else {
            // Default fallback
            gemm_nt<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, K, A, K, beta, C, M, stream);
        }
    } else if (lhs_format == 'C' && rhs_format == 'R') {
        // CR: cuBLAS uses opA=N, opB=T => C^T = B^T × A
        // CuTe: gemm_tn computes C^T = A × B^T with row-major A,B
        // After argument swap: gemm_tn(N, M, K, B, ldB, A, ldA, C, ldC)
        
        // Dispatch based on tile configuration
        // Note: gemm_tn has constraints due to GMMA layout requirements
        if (tile_M == 128 && tile_N == 128 && tile_K == 64) {
            gemm_tn<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else if (tile_M == 256 && tile_N == 128 && tile_K == 64) {
            gemm_tn<half_t, half_t, half_t, float, float, 256, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else if (tile_M == 128 && tile_N == 256 && tile_K == 64) {
            gemm_tn<half_t, half_t, half_t, float, float, 128, 256, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        } else {
            // Unsupported tile size for gemm_tn, fall back to default
            fprintf(stderr, "Warning: Tile size [%d,%d,%d] not supported by gemm_tn due to GMMA constraints, using default [128,128,64]\n", 
                    tile_M, tile_N, tile_K);
            gemm_tn<half_t, half_t, half_t, float, float, 128, 128, 64>(
                N, M, K, alpha, B, N, A, M, beta, C, M, stream);
        }
    } else {
        fprintf(stderr, "ERROR: Invalid storage format combination: lhs=%c, rhs=%c\n", 
                lhs_format, rhs_format);
    }
}

// Static cache for tile sizes to avoid repeated config file reads
static bool g_wgmma_tile_sizes_cached = false;
static int g_wgmma_cached_tile_M = 128;
static int g_wgmma_cached_tile_N = 128;
static int g_wgmma_cached_tile_K = 64;

/**
 * @brief WGMMA-based GEMM for FP16 with configurable tile sizes
 *
 * @param A_ptr Input matrix A
 * @param B_ptr Input matrix B
 * @param C_ptr Output matrix C
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 * @param lhs_format Storage format for A ('R' for row-major, 'C' for column-major)
 * @param rhs_format Storage format for B ('R' for row-major, 'C' for column-major)
 * @param stream CUDA stream
 */
void gemm_wgmma_fp16(const void* A_ptr, const void* B_ptr, void* C_ptr,
                     int M, int N, int K,
                     char lhs_format, char rhs_format,
                     cudaStream_t stream)
{
    // Load tile sizes from config only once (avoid repeated file I/O)
    if (!g_wgmma_tile_sizes_cached) {
        try {
            // Get configuration instance
            auto& config = kebab::config::ConfigParser::getInstance();

            // Get tile sizes from configuration
            auto tile_sizes = config.getOperatorTileSizes("gemm");

            if (tile_sizes.size() >= 3) {
                g_wgmma_cached_tile_M = tile_sizes[0];
                g_wgmma_cached_tile_N = tile_sizes[1];
                g_wgmma_cached_tile_K = tile_sizes[2];
            }
        } catch (const std::exception& e) {
            // Fallback to default tile sizes if config loading fails
            fprintf(stderr, "Warning: Failed to load tile sizes from config: %s\n", e.what());
            fprintf(stderr, "Using default tile sizes: 128x128x64\n");
        }
        g_wgmma_tile_sizes_cached = true;
    }

    // Call dispatch function with cached tile sizes
    gemm_wgmma_fp16_dispatch(A_ptr, B_ptr, C_ptr, M, N, K,
                            lhs_format, rhs_format,
                            g_wgmma_cached_tile_M, g_wgmma_cached_tile_N, g_wgmma_cached_tile_K, stream);
}

} // namespace cute
} // namespace kebab
