---
name: kebab-build-and-env
description: Build and bootstrap the Kebab project on Linux with CUDA, CMake, and Makefile workflows. Use when asked to set up dependencies, detect CUDA/GPU architecture, configure builds, or fix compilation environment issues.
---

# Kebab Build and Environment

## When to Use This Skill

- User asks to build project from scratch
- User sees CUDA detection failures (`CUDA_PATH` / `CUDA_ARCH` / `nvidia-smi`)
- User asks for debug/release build instructions
- User asks to clean and rebuild reliably

## Prerequisites

- Linux with NVIDIA driver and visible GPU (`nvidia-smi` works)
- CUDA toolkit and `nvcc`
- CMake (3.18+) and GNU Make
- `yaml-cpp` development headers

## Step-by-Step Workflows

### Workflow A: Standard Build (preferred)

1. Check environment:
   - `make gpu-info`
2. Prepare dependencies:
   - `make setup`
3. Build:
   - `make build`

### Workflow B: Explicit CMake Build

1. Configure:
   - `cmake -S kebab -B build -DCUDA_ARCH=sm_90a`
2. Compile:
   - `cmake --build build -j$(nproc)`

### Workflow C: Clean Rebuild

1. `make clean`
2. `make build`

## Environment Guidance

- If auto detection fails, set:
  - `export CUDA_PATH=/usr/local/cuda`
  - `export CUDA_ARCH=sm_90a`
- Hopper WGMMA workloads should use `sm_90a` when possible.

## Troubleshooting

| Issue | Mitigation |
|---|---|
| `GPU detection failed` from `Makefile` | Run `nvidia-smi`; then set `CUDA_ARCH` manually |
| `CUDA toolkit not found` | Set `CUDA_PATH` and ensure `${CUDA_PATH}/bin/nvcc` exists |
| CMake configure fails | Remove `build/` and rerun configure+build |

## References

- `Makefile` (`setup`, `gpu-info`, `cmake-configure`, `cmake-build`, `clean`)
- `kebab/CMakeLists.txt`
