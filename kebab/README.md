# Kebab Library

High-performance GPU kernel library using NVIDIA CuTe and WGMMA for Hopper GPUs.

## Building

### Using CMake

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Using Makefile (from project root)

```bash
make build
make bench-gemm
make test
```

## Usage

```cpp
#include <kebab/kebab.h>

using namespace kebab;

// Perform GEMM: C = A * B
cute::gemm(d_A, d_B, d_C, M, N, K, "RR");
```

## Project Structure

```
kebab/
├── include/kebab/       # Public headers
│   ├── cute/            # CuTe implementations
│   ├── cuda/            # CUDA baselines
│   ├── utils/           # Utilities
│   └── config/          # Configuration
├── lib/                 # Implementation
│   ├── cute/            # CuTe kernels
│   ├── cuda/            # CUDA baselines
│   ├── common/          # Common code
│   ├── benchmark/       # Benchmarks
│   └── examples/        # Examples
└── CMakeLists.txt       # Build config
```

## Documentation

See [BUILD.md](BUILD.md) for detailed build instructions.
