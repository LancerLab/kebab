/**
 * @file simple_gemm.cu
 * @brief Simple GEMM usage example
 */

#include <kebab/kebab.h>
#include <iostream>
#include <vector>

int main() {
    using namespace kebab;
    
    // Matrix dimensions
    const int M = 128, N = 128, K = 128;
    
    // Allocate host memory
    std::vector<__half> h_A(M * K), h_B(K * N), h_C(M * N);
    
    // Initialize matrices
    std::mt19937 gen(42);
    utils::initializeMatrix(h_A.data(), M, K, utils::InitMode::ONE, gen, 'R');
    utils::initializeMatrix(h_B.data(), K, N, utils::InitMode::ROW, gen, 'R');
    
    // Allocate device memory
    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));
    
    // Copy to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    
    // Perform GEMM: C = A × B (both row-major)
    cute::gemm(d_A, d_B, d_C, M, N, K, "RR");
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
    
    // Print result (first few elements)
    std::cout << "GEMM completed successfully!" << std::endl;
    std::cout << "First few elements of result:" << std::endl;
    for (int i = 0; i < std::min(5, M); ++i) {
        for (int j = 0; j < std::min(5, N); ++j) {
            std::cout << __half2float(h_C[i * N + j]) << " ";
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
