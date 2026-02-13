---
name: kebab-ncu-profiling-and-report
description: Profile Kebab kernels with Nsight Compute and export readable reports. Use when asked to run tune/mtune targets, collect NCU reports, convert .ncu-rep to text summaries, or troubleshoot performance bottlenecks.
---

# Kebab NCU Profiling and Report

## When to Use This Skill

- User asks for Nsight Compute profiling of GEMM/operator/microbench kernels
- User asks to produce report artifacts from `.ncu-rep`
- User asks for bottleneck diagnosis from existing profiling outputs

## Prerequisites

- `ncu` (or `nv-nsight-cu-cli`) available
- Built binaries (`make build`)

## Step-by-Step Workflows

### Workflow A: Operator Profiling

1. Build:
   - `make build`
2. Profile one operator path:
   - `make tune-gemm-cute`
   - `make tune-gemm-cuda`
   - `make tune-gemm-ref`
3. Profile all operators:
   - `make tune-all`

### Workflow B: Microbenchmark Profiling

1. Profile one microbenchmark:
   - `make mtune-copy-gmem-to-smem`
2. Profile all microbenchmarks:
   - `make mtune-all`

### Workflow C: Export Reports

1. Operator reports:
   - `make report`
2. Microbenchmark reports:
   - `make mreport`

Artifacts:

- Raw profiles in `profiling/*.ncu-rep`
- Text reports in `reports/*.txt`

## Optional Scripted Profiling

- `scripts/profile_gemm.sh` provides richer metric/section collection and exports:
  - `gemm_profile.ncu-rep`
  - `gemm_summary.txt`
  - `gemm_metrics.csv`

## Troubleshooting

| Issue | Mitigation |
|---|---|
| `ncu` not found | Install Nsight Compute or export path to binary |
| No `.ncu-rep` generated | Verify target binary path and GPU runtime permissions |
| Report conversion fails | Re-check `ncu --import` availability and source file existence |

## References

- `Makefile` (`tune-*`, `mtune-*`, `report`, `mreport`)
- `scripts/profile_gemm.sh`
- `profiling/` and `reports/`
