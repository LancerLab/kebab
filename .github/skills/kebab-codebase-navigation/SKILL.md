---
name: kebab-codebase-navigation
description: Navigate and modify the Kebab CUDA codebase safely. Use when asked to locate GEMM, CuTe, CUDA baseline, benchmark, microbenchmark, profiling, or build logic in this repository, and when deciding where a change should be implemented.
---

# Kebab Codebase Navigation

## When to Use This Skill

- User asks where to change GEMM or elementwise kernels
- User asks where build flags, CUDA architecture, or CMake targets are defined
- User asks where benchmark/microbenchmark binaries are wired
- User asks where profiling/reporting scripts live

## Project Map

- Main CMake entry: `kebab/CMakeLists.txt`
- Library target wiring: `kebab/lib/CMakeLists.txt`
- Benchmark executables: `kebab/lib/benchmark/CMakeLists.txt`
- Microbenchmark executables: `kebab/lib/microbench/CMakeLists.txt`
- Top-level workflow automation: `Makefile`
- Runtime/config knobs: `config.yaml`
- Performance docs and design notes: `docs/`

## Step-by-Step Workflow

1. Determine requested domain first: build, runtime kernel, benchmark, microbenchmark, profiling, or report.
2. For build/toolchain issues, inspect `Makefile` and `kebab/CMakeLists.txt` first.
3. For kernel implementation issues:
   - CuTe path: `kebab/lib/cute/`
   - CUDA baseline path: `kebab/lib/cuda/`
4. For executable mapping, verify target names in:
   - `kebab/lib/benchmark/CMakeLists.txt`
   - `kebab/lib/microbench/CMakeLists.txt`
5. Cross-check that proposed commands exist in `Makefile` target templates.

## Troubleshooting

| Issue | What to check |
|---|---|
| Command exists in docs but fails | Confirm target name in `Makefile` and generated binary path in `build/lib/...` |
| GEMM variant confusion | Verify CUDA vs CuTe file under `kebab/lib/cuda/` vs `kebab/lib/cute/` |
| Architecture mismatch | Inspect `CUDA_ARCH` handling in `Makefile` and `kebab/CMakeLists.txt` |

## References

- `kebab/CMakeLists.txt`
- `kebab/lib/CMakeLists.txt`
- `kebab/lib/benchmark/CMakeLists.txt`
- `kebab/lib/microbench/CMakeLists.txt`
- `Makefile`
