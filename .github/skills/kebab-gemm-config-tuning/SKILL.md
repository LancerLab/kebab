---
name: kebab-gemm-config-tuning
description: Tune GEMM experiment settings in Kebab using config.yaml and optimization docs. Use when asked to adjust matrix sizes, precision, layout modes, tile sizes, kernel versions, or to design reproducible performance experiments.
---

# Kebab GEMM Config Tuning

## When to Use This Skill

- User asks to change GEMM benchmark parameters
- User asks to compare layout modes (`RR`, `RC`, `CR`, `CC`)
- User asks to test precision (`float16`, `bfloat16`, `float32`) or kernel version lists
- User asks for tile-size and occupancy-oriented tuning plans

## Key Configuration Surface

Main file: `config.yaml`

Important keys under `operators.gemm`:

- `impl`
- `versions`
- `matrix_sizes`
- `modes`
- `precisions`
- `tile_sizes`
- `init_method`

## Step-by-Step Workflow

1. Edit `config.yaml` with one controlled change at a time.
2. Build and run benchmark:
   - `make build`
   - `make bench-gemm`
3. For deeper analysis, run:
   - `make tune-gemm-cute` (or cuda/ref)
4. Save and compare outputs from `bench_results/`, `profiling/`, and `reports/`.

## Tuning Rules from Repository Context

- Hopper/WGMMA paths generally expect `sm_90a`.
- The documented tile constraint notes K dimension compatibility requirements for GMMA pathways.
- Keep experiment dimensions and precision explicit to ensure reproducibility.

## Troubleshooting

| Issue | Mitigation |
|---|---|
| Unexpected perf regression | Revert to previous `versions`/`tile_sizes` baseline and isolate one variable |
| Precision mismatch | Verify selected kernel path supports requested precision |
| Layout-specific anomaly | Test all `modes` and compare independently |

## References

- `config.yaml`
- `docs/optimization_plan.md`
- `docs/TILE_SIZE_CONFIGURATION.md`
- `Makefile` GEMM benchmark/profile targets
