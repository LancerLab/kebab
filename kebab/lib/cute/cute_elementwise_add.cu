/**
 * @file elementwise_add.cu
 * @brief CuTe-style element-wise addition implementation
 * 
 * Uses CuTe's Layout and Tensor abstractions for clean, type-safe code.
 */

#include "kebab/cute/elementwise_add.h"
#include <cute/tensor.hpp>

namespace kebab {
namespace cute {

using namespace ::cute;  // Use global cute namespace

/**
 * @brief CuTe element-wise addition kernel
 */
template<typename T>
__global__ void elementwise_add_kernel_cute(
    const T* A_ptr, const T* B_ptr, T* C_ptr, int N)
{
    // Create CuTe layout and tensors
    auto layout = make_layout(make_shape(N), make_stride(Int<1>{}));
    
    Tensor gA = make_tensor(make_gmem_ptr(A_ptr), layout);
    Tensor gB = make_tensor(make_gmem_ptr(B_ptr), layout);
    Tensor gC = make_tensor(make_gmem_ptr(C_ptr), layout);
    
    // Grid-stride loop
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < size(gC); 
         i += blockDim.x * gridDim.x) {
        gC(i) = gA(i) + gB(i);
    }
}

/**
 * @brief Vectorized kernel using CuTe array
 */
template<typename T, int VecWidth>
__global__ void elementwise_add_kernel_cute_vec(
    const T* A_ptr, const T* B_ptr, T* C_ptr, int N)
{
    int N_vec = N / VecWidth;
    
    auto layout_vec = make_layout(make_shape(N_vec), make_stride(Int<1>{}));
    
    using VecType = cute::array<T, VecWidth>;
    Tensor gA_vec = make_tensor(make_gmem_ptr(reinterpret_cast<const VecType*>(A_ptr)), layout_vec);
    Tensor gB_vec = make_tensor(make_gmem_ptr(reinterpret_cast<const VecType*>(B_ptr)), layout_vec);
    Tensor gC_vec = make_tensor(make_gmem_ptr(reinterpret_cast<VecType*>(C_ptr)), layout_vec);
    
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < size(gC_vec); 
         i += blockDim.x * gridDim.x) {
        VecType a = gA_vec(i);
        VecType b = gB_vec(i);
        VecType c;
        
        #pragma unroll
        for (int j = 0; j < VecWidth; ++j) {
            c[j] = a[j] + b[j];
        }
        
        gC_vec(i) = c;
    }
    
    // Remainder
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = N_vec * VecWidth; i < N; ++i) {
            C_ptr[i] = A_ptr[i] + B_ptr[i];
        }
    }
}

/**
 * @brief Host function
 */
template<typename T>
void elementwise_add_impl(const T* A, const T* B, T* C, int N, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || C == nullptr) {
        fprintf(stderr, "ERROR: Null pointer in elementwise_add\n");
        return;
    }
    if (N <= 0) {
        fprintf(stderr, "ERROR: Invalid size N=%d\n", N);
        return;
    }
    
    constexpr int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);
    
    constexpr int VecWidth = (sizeof(T) == sizeof(float)) ? 4 : 8;
    
    if (N >= VecWidth && (N % VecWidth == 0)) {
        elementwise_add_kernel_cute_vec<T, VecWidth><<<blocks, threads, 0, stream>>>(A, B, C, N);
    } else {
        elementwise_add_kernel_cute<T><<<blocks, threads, 0, stream>>>(A, B, C, N);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

template<>
void elementwise_add<float>(const float* A, const float* B, float* C, int N, cudaStream_t stream) {
    elementwise_add_impl(A, B, C, N, stream);
}

template<>
void elementwise_add<__half>(const __half* A, const __half* B, __half* C, int N, cudaStream_t stream) {
    elementwise_add_impl(A, B, C, N, stream);
}

} // namespace cute
} // namespace kebab
