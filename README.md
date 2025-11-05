# Kebab - High-Performance GPU Kernel Library

A modern C++ library providing high-performance GPU kernels using NVIDIA CuTe and WGMMA instructions for Hopper architecture GPUs.

## Features

- **High-Performance GEMM**: Optimized matrix multiplication using WGMMA instructions
- **Multiple Storage Formats**: Support for row-major and column-major layouts
- **FP16 Precision**: Optimized for Tensor Core operations
- **Flexible Configuration**: YAML-based configuration system
- **Comprehensive Benchmarking**: Built-in performance testing and verification

## Prerequisites

- CUDA Toolkit 12.0+ (with Hopper SM90+ support)
- CMake 3.18+
- C++17 compatible compiler
- NVIDIA GPU with compute capability 9.0+ (Hopper architecture)

## Quick Start

### Build

```bash
# Configure
cmake -S kebab -B kebab/build

# Build
cmake --build kebab/build -j$(nproc)
```

### Run Benchmarks

```bash
# Using Makefile (recommended)
make bench-gemm
make bench-elementwise-add

# Or directly
kebab/build/lib/benchmark/bench_gemm
```

### Run Tests

```bash
make test
```

## Usage

```cpp
#include <kebab/kebab.h>

using namespace kebab;

// Initialize matrices
int M = 1024, N = 1024, K = 1024;
std::vector<__half> h_A(M * K), h_B(K * N), h_C(M * N);

// Allocate device memory
__half *d_A, *d_B, *d_C;
cudaMalloc(&d_A, M * K * sizeof(__half));
cudaMalloc(&d_B, K * N * sizeof(__half));
cudaMalloc(&d_C, M * N * sizeof(__half));

// Copy data to device
cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);

// Perform GEMM: C = A * B
cute::gemm(d_A, d_B, d_C, M, N, K, "RR");

// Copy result back
cudaMemcpy(h_C.data(), d_C, M * N * sizeof(__half), cudaMemcpyDeviceToHost);
```

## Project Structure

```
kebab/
├── include/kebab/           # Public API headers
│   ├── cute/                # CuTe implementations
│   ├── cuda/                # CUDA baselines
│   ├── utils/               # Utility functions
│   └── config/              # Configuration system
├── lib/                     # Implementation
│   ├── cute/                # CuTe kernels
│   ├── cuda/                # CUDA baselines
│   ├── common/              # Common code
│   ├── benchmark/           # Benchmarks
│   └── examples/            # Usage examples
└── CMakeLists.txt           # Build configuration
```

## Configuration

Edit `config.yaml` to customize benchmark parameters:

```yaml
benchmark:
  warmup_runs: 10
  measurement_runs: 100
  matrix_sizes: [256, 512, 1024]
```

## Troubleshooting

### CUDA Not Found

```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
```

### Driver Version Mismatch

Ensure your NVIDIA driver supports the CUDA toolkit version you're using.

### Build Errors

```bash
# Clean and rebuild
rm -rf kebab/build
cmake -S kebab -B kebab/build
cmake --build kebab/build -j$(nproc)
```

### GPU Not Detected

```bash
# Check GPU visibility
nvidia-smi

# Verify CUDA installation
nvcc --version
```

## Documentation

- [Build Guide](kebab/BUILD.md) - Detailed build instructions
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - Adding new operators
- [CuTe GEMM Implementation](docs/CUTE_GEMM_IMPLEMENTATION.md) - Implementation details

## License

[Specify your license here]
