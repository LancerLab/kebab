---
name: kebab-benchmark-and-run
description: Run operator benchmarks and single-run executables in Kebab for GEMM and elementwise add. Use when asked to execute performance baselines, compare implementations, or run quick correctness/perf checks through Make targets.
---

# Kebab Benchmark and Run

## When to Use This Skill

- User asks to benchmark GEMM or elementwise add
- User asks to run one implementation quickly (cute/cuda/ref)
- User asks to generate benchmark summary CSV/Markdown outputs

## Prerequisites

- Build succeeded (`make build`)
- GPU and CUDA available

## Step-by-Step Workflows

### Workflow A: Operator Benchmarks

1. Build:
   - `make build`
2. Run one benchmark:
   - `make bench-gemm`
   - `make bench-elementwise-add`
3. Run all benchmarks:
   - `make bench-all`

Expected outputs:

- CSV files under `bench_results/`
- Optional summary: `bench_results/summary.md`

### Workflow B: Single-Run Binaries

For GEMM variants:

- `make run-gemm-cute`
- `make run-gemm-cuda`
- `make run-gemm-ref`

## Notes

- `make test` maps to `bench-gemm` in this repository.
- Elementwise single-run binary naming in CMake differs from `run-*-cute` template usage; prefer `bench-elementwise-add` for reliable checks.

## Troubleshooting

| Issue | Mitigation |
|---|---|
| Benchmark target not found | Confirm target exists in `Makefile` help output |
| Binary missing under `build/lib/benchmark` | Re-run `make build` and inspect CMake benchmark targets |
| Empty summary report | Verify CSV files exist and `scripts/generate_report.py` is present |

## References

- `Makefile` (`bench-*`, `bench-all`, `run-*`, `test`)
- `kebab/lib/benchmark/CMakeLists.txt`
- `scripts/generate_report.py`
