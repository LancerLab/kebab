---
name: kebab-microbench-and-dump
description: Execute Kebab microbenchmarks and generate PTX/SASS dumps for kernel-level analysis. Use when asked to run mbench targets, inspect assembly output, compare copy or MMA kernels, or analyze low-level code generation.
---

# Kebab Microbenchmark and Dump

## When to Use This Skill

- User requests microbenchmark execution (`mbench-*`)
- User needs SASS/PTX dumps for kernels
- User wants to compare implementations like native/vectorized/PTX/CuTe copy paths

## Prerequisites

- Successful build (`make build`)
- `cuobjdump` available in CUDA toolkit

## Step-by-Step Workflows

### Workflow A: Run Microbenchmark

1. Build:
   - `make build`
2. Run one microbenchmark:
   - `make mbench-copy-gmem-to-smem`
   - `make mbench-mma-wgmma`
   - `make mbench-hgemm`

### Workflow B: Generate Kernel Dumps

1. Build:
   - `make build`
2. Dump one microbenchmark binary:
   - `make mdump-copy-gmem-to-smem`
3. Inspect outputs under:
   - `dump/microbench/<microbench_name>/`

### Workflow C: Operator Dumps

- `make dump-gemm-cute`
- `make dump-gemm-cuda`
- `make dump-gemm-ref`

Outputs are under `dump/operator/`.

## Troubleshooting

| Issue | Mitigation |
|---|---|
| `cuobjdump not found` | Ensure CUDA toolkit bin directory is available and retry |
| No PTX generated | Some binaries may not embed PTX; SASS output is still usable |
| Dump files too large | Focus on split kernel files instead of `all_kernels.*` |

## References

- `Makefile` (`mbench-*`, `mdump-*`, `dump-*-cute/cuda/ref`)
- `kebab/lib/microbench/CMakeLists.txt`
- `dump/`
