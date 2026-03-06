#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <iostream>

__global__ void probe_nontrans(float* out_matrix) {
  __shared__ __align__(16) __half smem[64 * 128];

  for (int i = threadIdx.x; i < 64 * 128; i += blockDim.x) {
    smem[i] = __float2half(-1.0f);
  }
  __syncthreads();

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  __half vals[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    vals[k] = __float2half(float(lane * 8 + k + 1));
  }

  int* regs = reinterpret_cast<int*>(vals);

  constexpr int N = 128;
  uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t tile_base = base + static_cast<uint32_t>((warp * 16 * N) * sizeof(__half));
  uint32_t lane_offset = static_cast<uint32_t>((lane % 8) * N + (lane / 16) * N * 8 + (lane & 8));
  uint32_t addr = tile_base + lane_offset * sizeof(__half);

  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));

  __syncthreads();

  for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
    int r = i / 16;
    int c = i % 16;
    out_matrix[i] = __half2float(smem[r * N + c]);
  }
}

int main() {
  float* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(float) * 16 * 16);
  probe_nontrans<<<1, 32>>>(d_out);
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "kernel error: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  std::vector<float> h(16 * 16);
  cudaMemcpy(h.data(), d_out, sizeof(float) * 16 * 16, cudaMemcpyDeviceToHost);

  int mismatch = 0;
  for (int lane = 0; lane < 32; ++lane) {
    int row0 = lane / 4;
    int row1 = row0 + 8;
    int col0 = (lane % 4) * 2;
    int col1 = col0 + 1;
    int base = lane * 8 + 1;

    int got[8] = {
      (int)h[row0 * 16 + col0],
      (int)h[row0 * 16 + col1],
      (int)h[row1 * 16 + col0],
      (int)h[row1 * 16 + col1],
      (int)h[row0 * 16 + (col0 + 8)],
      (int)h[row0 * 16 + (col1 + 8)],
      (int)h[row1 * 16 + (col0 + 8)],
      (int)h[row1 * 16 + (col1 + 8)],
    };
    for (int k = 0; k < 8; ++k) {
      if (got[k] != base + k) mismatch++;
    }

    if (lane < 8) {
      std::cout << "lane " << lane << ": ";
      for (int k = 0; k < 8; ++k) {
        int v = got[k];
        int src_lane = (v > 0) ? ((v - 1) / 8) : -1;
        int src_idx = (v > 0) ? ((v - 1) % 8) : -1;
        std::cout << "[k" << k << "<-l" << src_lane << "." << src_idx << "] ";
      }
      std::cout << "\n";
    }
  }

  std::cout << "mismatch=" << mismatch << "\n";
  cudaFree(d_out);
  return 0;
}

