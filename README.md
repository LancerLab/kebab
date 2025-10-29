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
make bench-elementwise_add
```

Or run all benchmarks and generate a summary report:

```bash
make bench-all
```

Results are saved to:
- `bench_results/<operator>_results.csv` - Raw benchmark data
- `bench_results/summary.md` - Aggregated performance summary

### 4. Profile with Nsight Compute

Profile a specific operator:

```bash
make tune-elementwise_add
```

Or profile all operators:

```bash
make tune-all
```

Profiling outputs are saved to:
- `profiling/<operator>_profile.ncu-rep` - Binary NCU report (open in Nsight Compute GUI)
- `profiling/<operator>_summary.txt` - Text summary with key metrics

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

### Iteration 2: GEMM (Planned)
- Matrix multiplication with Tensor Cores
- Software pipelining with async copy
- Target: ≥90% of cuBLAS performance

### Iteration 3: Additional Operator (Planned)
- Convolution or Reduction operator

## Performance Targets

All operators target state-of-the-art performance:
- **Tensor Core Utilization**: Use mma.sync/MMA atoms for compute-intensive ops
- **Memory Optimization**: Async copy, TMA, coalesced access patterns
- **Target Performance**: ≥90% of vendor library performance (cuBLAS, cuDNN)

## Troubleshooting

### GPU Detection Fails

**Error**: `GPU detection failed`

**Solution**: 
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Ensure GPU is visible to the system
- Check CUDA_VISIBLE_DEVICES environment variable

### CuTe Clone Fails

**Error**: `Failed to clone CuTe`

**Solution**:
- Check network connection
- Verify git is installed
- Try manual clone: `git clone https://github.com/NVIDIA/cutlass.git third_party/cute/`

### Compilation Errors

**Error**: Architecture-specific compilation failures

**Solution**:
- Verify CUDA toolkit version: `nvcc --version`
- Check GPU compute capability matches CUDA version
- Override auto-detection in config.yaml: `cuda_arch: sm_80`

### yaml-cpp Not Found

**Error**: `yaml-cpp library not found`

**Solution**:
- Install via package manager: `sudo apt-get install libyaml-cpp-dev` (Ubuntu/Debian)
- Or build from source during `make setup`

## Development

See the [Developer Guide](docs/DEVELOPER_GUIDE.md) for:
- Adding new operators
- Extending the build system
- Writing benchmarks
- Profiling and optimization workflows

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please follow the established patterns for operators, baselines, and benchmarks.

## Acknowledgments

- NVIDIA CUTLASS team for the CuTe template library
- CUDA toolkit and Nsight Compute for profiling capabilities
