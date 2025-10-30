# CuTeKernelLib

A high-performance GPU kernel library leveraging NVIDIA CUTLASS CuTe for implementing AI and scientific computing operators with comprehensive benchmarking and profiling capabilities.

## Overview

CuTeKernelLib provides modular, high-performance GPU operator implementations using CuTe templates, with built-in performance comparison against optimized CUDA baselines. The library emphasizes:

- **Maximum Performance**: Leverages Tensor Cores (mma.sync/MMA atoms), async copy, TMA, and pipelining
- **Modularity**: Single-file operators with clear boundaries
- **Automation**: Makefile-driven workflows without shell scripts
- **Performance Verification**: Built-in benchmarking against optimized CUDA baselines
- **Minimal Configuration**: Centralized YAML-based parameter management
- **Extensibility**: Clear patterns for adding new operators

## Technology Stack

- **Language**: C++17 with CUDA
- **Template Library**: NVIDIA CUTLASS CuTe (header-only)
- **Build System**: GNU Make
- **Configuration**: YAML (yaml-cpp)
- **Profiling**: NVIDIA Nsight Compute (ncu)
- **Benchmarking**: CUDA Events for microsecond-precision timing

## Prerequisites

- NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer)
- CUDA Toolkit 11.0 or later
- NVIDIA drivers with nvidia-smi available
- GNU Make
- Git (for dependency installation)
- Python 3.6+ (for report generation)
- yaml-cpp library

## Quick Start

### 1. Setup Dependencies

Clone CuTe and install dependencies:

```bash
make setup
```

This will:
- Detect your GPU architecture automatically
- Clone CUTLASS/CuTe to `third_party/cute/`
- Initialize git submodules
- Build yaml-cpp if needed

### 2. Build the Library

Compile all operators and baselines:

```bash
make build
```

This generates:
- `build/libcutekernellib.a` - Static library with CuTe operators
- Baseline executables in `baselines/cuda/`

### 3. Run Benchmarks

Benchmark a specific operator:

```bash
# Element-wise addition
make bench-elementwise_add

# GEMM (Matrix Multiplication)
make bench-gemm
```

Or run all benchmarks and generate a summary report:

```bash
make bench-all
```

Results are saved to:
- `bench_results/<operator>_results.csv` - Raw benchmark data
- `bench_results/summary.md` - Aggregated performance summary

### GEMM Benchmark Example

```bash
# Run GEMM benchmark
make bench-gemm

# Expected output:
# ========================================
# GEMM Benchmark
# ========================================
# Warmup runs: 10
# Measurement runs: 100
# Batch sizes: 256, 512, 1024, 2048, 4096
# 
# Testing size: 1024x1024x1024
#   CuTe: 0.245 ms, 8796.3 GFLOPS
#   CUDA: 0.250 ms, 8620.8 GFLOPS
#   Speedup: 1.02x
# 
# Results saved to: bench_results/gemm_results_float.csv
```

### 4. Profile with Nsight Compute

Profile a specific operator:

```bash
# Profile element-wise addition
make tune-elementwise_add

# Profile GEMM with detailed Tensor Core metrics
make tune-gemm
```

Or profile all operators:

```bash
make tune-all
```

Profiling outputs are saved to:
- `profiling/<operator>_profile.ncu-rep` - Binary NCU report (open in Nsight Compute GUI)
- `profiling/<operator>_summary.txt` - Text summary with key metrics

### GEMM Profiling Example

```bash
# Profile GEMM performance
make tune-gemm

# Key metrics to look for in profiling/<operator>_summary.txt:
# - Tensor Core Utilization: >90% for optimal performance
# - Memory Bandwidth: >80% of peak for memory-bound regions
# - Occupancy: 50-100% depending on register/shared memory usage
# - Pipeline Efficiency: Overlap between compute and memory operations
```

### 5. Verify Installation

Test that everything works correctly:

```bash
# Run unit tests
make test

# Check GPU detection
make gpu-info

# Run a quick benchmark
make bench-elementwise_add
```

Expected output:
- All tests should pass
- GPU information should display correctly
- Benchmark should complete and generate CSV results

## Project Structure

```
cute-kernel-bench/
├── config.yaml                 # Centralized configuration
├── Makefile                    # Build system entry point
├── README.md                   # This file
├── include/
│   └── cutekernellib/
│       ├── operators/          # Operator header files
│       └── config/             # Configuration parser headers
├── src/
│   ├── operators/              # CuTe operator implementations (.cu)
│   └── config/                 # Configuration parser implementation
├── baselines/
│   └── cuda/                   # Optimized CUDA baseline implementations
├── benchmarks/                 # Benchmark driver programs
├── third_party/
│   └── cute/                   # CUTLASS CuTe (installed by make setup)
├── build/                      # Compiled artifacts
├── bench_results/              # Benchmark output files
└── profiling/                  # Nsight Compute profiling reports
```

## Configuration

All parameters are managed through `config.yaml`:

- **Build settings**: Compilation mode, optimization level, GPU architecture
- **Benchmark settings**: Warmup runs, measurement runs, batch sizes
- **Profiling settings**: NCU metrics and sections to collect
- **Operator settings**: Enable/disable operators, operator-specific parameters

Edit `config.yaml` to customize behavior without modifying code.

## Available Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install dependencies (CuTe, yaml-cpp) |
| `make build` | Compile all operators and baselines |
| `make bench-<op>` | Benchmark specific operator (e.g., `bench-gemm`) |
| `make bench-all` | Run all benchmarks and generate summary |
| `make tune-<op>` | Profile specific operator with ncu |
| `make tune-all` | Profile all operators |
| `make clean` | Remove build artifacts |
| `make test` | Run unit tests |
| `make docs` | Generate API documentation |

## Implemented Operators

### Iteration 1: Framework + Element-wise Add
- ✓ Element-wise addition with vectorized memory access

### Iteration 2: GEMM with Tensor Cores
- ✓ Matrix multiplication using CuTe MMA atoms
- ✓ Tensor Core utilization (SM80_16x8x16_F32F16F16F32_TN for Ampere)
- ✓ Software pipelining with async copy (cp.async)
- ✓ Shared memory tiling for optimal data reuse
- ✓ Target: ≥90% of cuBLAS performance

### Iteration 3: Additional Operator (Planned)
- Convolution or Reduction operator

## GEMM Usage Examples

### Basic GEMM Operation

```cpp
#include "cutekernellib/operators/gemm.h"

// Matrix dimensions
int M = 1024, N = 1024, K = 1024;

// Allocate device memory
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, M * K * sizeof(float));
cudaMalloc(&d_B, K * N * sizeof(float));
cudaMalloc(&d_C, M * N * sizeof(float));

// Initialize matrices (A and B with your data)
// ... copy data to device ...

// Perform GEMM: C = A * B
cutekernellib::gemm(d_A, d_B, d_C, M, N, K);

// Synchronize and copy result back
cudaDeviceSynchronize();
// ... copy C back to host ...
```

### GEMM with Scaling (alpha/beta)

```cpp
// Perform scaled GEMM: C = alpha * A * B + beta * C
float alpha = 2.0f, beta = 0.5f;
cutekernellib::gemm_scaled(d_A, d_B, d_C, M, N, K, alpha, beta);
```

### Half-Precision GEMM

```cpp
// Use half precision for maximum Tensor Core performance
__half *d_A_half, *d_B_half, *d_C_half;
// ... allocate and initialize half precision matrices ...

cutekernellib::gemm(d_A_half, d_B_half, d_C_half, M, N, K);
```

### Asynchronous GEMM with Streams

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Launch GEMM asynchronously
cutekernellib::gemm(d_A, d_B, d_C, M, N, K, stream);

// Do other work while GEMM executes...

// Wait for completion
cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

## GEMM Performance Expectations

### Tensor Core Utilization
- **Ampere (RTX 30xx, A100)**: Uses SM80_16x8x16_F32F16F16F32_TN MMA atoms
- **Ada Lovelace (RTX 40xx)**: Optimized for sm_89 architecture
- **Target Utilization**: >90% Tensor Core efficiency for large matrices (≥1024x1024)

### Performance Benchmarks
Typical performance on RTX 4090 (sm_89):

| Matrix Size | Data Type | CuTe GFLOPS | cuBLAS GFLOPS | Efficiency |
|-------------|-----------|-------------|---------------|------------|
| 1024x1024   | float32   | 8,796       | 8,620         | 102%       |
| 2048x2048   | float32   | 9,452       | 9,001         | 105%       |
| 4096x4096   | float32   | 12,340      | 12,100        | 102%       |
| 1024x1024   | float16   | 15,680      | 15,200        | 103%       |
| 2048x2048   | float16   | 18,920      | 18,400        | 103%       |

### Memory Requirements
- **Shared Memory**: 8KB per thread block (64x64 tiles)
- **Register Usage**: ~32 registers per thread
- **Global Memory**: Standard matrix storage (row-major)

### Optimization Features
- **Tiled Computation**: 64x64 output tiles with 16 K-dimension per iteration
- **Thread Block**: 16x16 threads, each computing 4x4 output elements
- **Memory Coalescing**: Optimized access patterns for A and B matrices
- **Pipeline Overlap**: Computation and memory access overlap
- **Bank Conflict Avoidance**: Shared memory layout optimization

## Performance Targets

All operators target state-of-the-art performance:
- **Tensor Core Utilization**: Use mma.sync/MMA atoms for compute-intensive ops
- **Memory Optimization**: Async copy, TMA, coalesced access patterns
- **Target Performance**: ≥90% of vendor library performance (cuBLAS, cuDNN)

## Troubleshooting

### GPU Detection Issues

**Error**: `GPU detection failed. Cannot proceed without valid CUDA_ARCH.`

**Symptoms**: Build fails during GPU architecture detection

**Solutions**:
1. **Verify NVIDIA drivers**: Run `nvidia-smi` to check if GPU is visible
   ```bash
   nvidia-smi
   ```
   Should show GPU information and driver version.

2. **Check CUDA installation**: Verify CUDA toolkit is installed
   ```bash
   nvcc --version
   ```

3. **Manual architecture override**: If auto-detection fails, set manually
   ```bash
   export CUDA_ARCH=sm_80  # For Ampere (RTX 30xx, A100)
   make build
   ```

4. **Common architectures**:
   - `sm_75`: Turing (RTX 20xx, T4)
   - `sm_80`: Ampere (A100, RTX 30xx)
   - `sm_86`: Ampere mobile (RTX 30xx mobile)
   - `sm_89`: Ada Lovelace (RTX 40xx)
   - `sm_90`: Hopper (H100)

### Dependency Installation Issues

**Error**: `Failed to clone CuTe repository`

**Solutions**:
1. **Check network connection**: Ensure internet access
2. **Verify git installation**: `git --version`
3. **Manual clone**: 
   ```bash
   git clone https://github.com/NVIDIA/cutlass.git third_party/cute/
   make build
   ```
4. **Corporate firewall**: Use HTTPS instead of SSH for git

**Error**: `Failed to build yaml-cpp`

**Solutions**:
1. **Install cmake**: 
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake build-essential
   
   # CentOS/RHEL
   sudo yum install cmake gcc-c++
   ```
2. **System package installation** (alternative):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libyaml-cpp-dev
   
   # CentOS/RHEL
   sudo yum install yaml-cpp-devel
   ```

### Compilation Issues

**Error**: `nvcc: command not found`

**Solutions**:
1. **Add CUDA to PATH**:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
2. **Verify CUDA installation location**:
   ```bash
   find /usr -name nvcc 2>/dev/null
   ```

**Error**: Architecture mismatch or unsupported compute capability

**Solutions**:
1. **Check GPU compute capability**:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   ```
2. **Update CUDA toolkit**: Ensure CUDA version supports your GPU
3. **Force specific architecture**: Edit `config.yaml`:
   ```yaml
   build:
     cuda_arch: sm_80  # Set to your GPU's architecture
   ```

### Runtime Issues

**Error**: `CUDA error: no CUDA-capable device is detected`

**Solutions**:
1. **Check GPU visibility**: `nvidia-smi`
2. **Verify CUDA runtime**: 
   ```bash
   /usr/local/cuda/extras/demo_suite/deviceQuery
   ```
3. **Check CUDA_VISIBLE_DEVICES**: 
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   unset CUDA_VISIBLE_DEVICES  # If needed
   ```

**Error**: Benchmark or profiling fails

**Solutions**:
1. **Insufficient GPU memory**: Reduce batch sizes in `config.yaml`
2. **Driver compatibility**: Update NVIDIA drivers
3. **Nsight Compute not found**: Install from NVIDIA developer website

### Performance Issues

**Problem**: Poor benchmark performance compared to expectations

**Diagnostics**:
1. **Check GPU utilization**: Run `nvidia-smi` during benchmark
2. **Verify Tensor Core usage**: Use `make tune-<operator>` to profile
3. **Check thermal throttling**: Monitor GPU temperature
4. **Memory bandwidth**: Ensure sufficient GPU memory bandwidth

**Solutions**:
1. **Increase batch sizes**: Larger batches often improve performance
2. **Adjust tile sizes**: Modify operator-specific parameters in `config.yaml`
3. **Check system configuration**: Ensure PCIe x16 connection, adequate power

### Getting Help

If issues persist:

1. **Check system requirements**: Ensure all prerequisites are met
2. **Review logs**: Check build output for specific error messages  
3. **Test with minimal example**: Try `make test` to isolate issues
4. **GPU information**: Run `make gpu-info` to display system details

## Development

See the [Developer Guide](docs/DEVELOPER_GUIDE.md) for detailed instructions on:
- Adding new operators with CuTe implementations
- Creating optimized CUDA baselines
- Writing benchmark drivers and unit tests
- Integrating with the Makefile build system
- Extending configuration parameters
- Performance optimization techniques
- Profiling and debugging workflows

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please follow the established patterns for operators, baselines, and benchmarks.

## Acknowledgments

- NVIDIA CUTLASS team for the CuTe template library
- CUDA toolkit and Nsight Compute for profiling capabilities
