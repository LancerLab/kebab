#!/usr/bin/env python3
"""
GEMM autotuning script for Kebab.

This script explores a hyperparameter grid over currently strong CUDA GEMM kernels
and reports the best-performing configurations.

Tunable dimensions:
- CUDA kernel version (e.g., 15/18/19/20)
- matrix_size
- mode (e.g., RC)
- precision (e.g., float16)
- v20 decomposition mode via env KEBAB_V20_MODE (heuristic/dp/persistent)

Outputs:
- JSON summary under bench_results/
- Markdown report under bench_results/
- Optional config.yaml update with recommended ordering
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


@dataclass
class Trial:
    version: int
    mode: str
    precision: str
    matrix_size: int
    v20_mode: Optional[str]
    variant: str
    latency_ms: float
    tflops: float
    speedup: float
    est_cublas_tflops: float


def parse_list_int(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_list_str(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML root in {path}")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def ensure_gemm_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "operators" not in cfg or "gemm" not in cfg["operators"]:
        raise RuntimeError("config.yaml missing operators.gemm")
    return cfg["operators"]["gemm"]


def run_trial(
    repo_root: Path,
    bench_bin: Path,
    cfg_path: Path,
    base_cfg: Dict[str, Any],
    version: int,
    mode: str,
    precision: str,
    matrix_size: int,
    warmup_runs: int,
    measurement_runs: int,
    v20_mode: Optional[str],
    timeout_s: int,
) -> Trial:
    cfg = json.loads(json.dumps(base_cfg))

    if "benchmark" not in cfg:
        cfg["benchmark"] = {}
    cfg["benchmark"]["warmup_runs"] = warmup_runs
    cfg["benchmark"]["measurement_runs"] = measurement_runs

    gemm_cfg = ensure_gemm_config(cfg)
    gemm_cfg["enabled"] = True
    gemm_cfg["impl"] = ["cuda"]
    gemm_cfg["versions"] = [version]
    gemm_cfg["modes"] = [mode]
    gemm_cfg["precisions"] = [precision]
    gemm_cfg["matrix_sizes"] = [matrix_size]

    save_yaml(cfg_path, cfg)

    env = os.environ.copy()
    if v20_mode:
        env["KEBAB_V20_MODE"] = v20_mode
    else:
        env.pop("KEBAB_V20_MODE", None)

    run = subprocess.run(
        [str(bench_bin)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(
            f"bench_gemm failed (version={version}, mode={mode}, precision={precision}, size={matrix_size}, v20_mode={v20_mode}).\n"
            f"stdout:\n{run.stdout}\n\nstderr:\n{run.stderr}"
        )

    csv_path = repo_root / f"bench_results/gemm_results_{precision}_cuda.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Expected CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"CSV has no rows: {csv_path}")

    best_row: Optional[Dict[str, str]] = None
    best_tflops = -1.0
    tag = f"cuda_v{version}_"
    for row in rows:
        variant = row.get("Variant", "")
        if not variant.startswith(tag):
            continue
        tflops = float(row.get("Throughput(GFLOPS)", "0")) / 1000.0
        if tflops > best_tflops:
            best_tflops = tflops
            best_row = row

    if best_row is None:
        raise RuntimeError(f"No CSV row matched variant prefix {tag} in {csv_path}")

    latency_ms = float(best_row["Latency(ms)"])
    throughput_gflops = float(best_row["Throughput(GFLOPS)"])
    speedup = float(best_row["Speedup"])
    tflops = throughput_gflops / 1000.0
    est_cublas_tflops = (tflops / speedup) if speedup > 0 else 0.0

    return Trial(
        version=version,
        mode=mode,
        precision=precision,
        matrix_size=matrix_size,
        v20_mode=v20_mode,
        variant=best_row["Variant"],
        latency_ms=latency_ms,
        tflops=tflops,
        speedup=speedup,
        est_cublas_tflops=est_cublas_tflops,
    )


def trial_sort_key(trial: Trial):
    return (trial.speedup, trial.tflops, -trial.latency_ms)


def write_reports(repo_root: Path, trials: List[Trial], top_k: int) -> Dict[str, Path]:
    bench_dir = repo_root / "bench_results"
    bench_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ranked = sorted(trials, key=trial_sort_key, reverse=True)
    top = ranked[:top_k]

    json_path = bench_dir / f"autotune_gemm_{ts}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "num_trials": len(trials),
                "top_k": top_k,
                "best": asdict(top[0]) if top else None,
                "trials": [asdict(t) for t in ranked],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    md_path = bench_dir / f"autotune_gemm_{ts}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GEMM Autotune Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Trials: {len(trials)}\n\n")
        f.write("| Rank | Version | v20_mode | Mode | Precision | Size | TFLOPS | Speedup | Latency(ms) | Variant |\n")
        f.write("|---:|---:|---|---|---|---:|---:|---:|---:|---|\n")
        for idx, t in enumerate(top, start=1):
            f.write(
                f"| {idx} | {t.version} | {t.v20_mode or '-'} | {t.mode} | {t.precision} | {t.matrix_size} | "
                f"{t.tflops:.3f} | {t.speedup:.3f} | {t.latency_ms:.4f} | {t.variant} |\n"
            )
        f.write("\n")
        if top:
            best = top[0]
            f.write("## Recommended config snippets\n\n")
            f.write(f"- versions priority: [{best.version}]\n")
            f.write(f"- mode: [{best.mode}]\n")
            f.write(f"- precision: [{best.precision}]\n")
            f.write(f"- matrix_sizes: [{best.matrix_size}]\n")
            if best.version == 20:
                f.write(f"- export KEBAB_V20_MODE={best.v20_mode or 'heuristic'}\n")

    return {"json": json_path, "md": md_path}


def apply_best_to_config(cfg_path: Path, base_cfg: Dict[str, Any], best: Trial) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    gemm_cfg = ensure_gemm_config(cfg)

    current_versions = gemm_cfg.get("versions", [])
    if not isinstance(current_versions, list):
        current_versions = [current_versions]

    new_versions = [best.version] + [v for v in current_versions if v != best.version]
    gemm_cfg["versions"] = new_versions
    gemm_cfg["modes"] = [best.mode]
    gemm_cfg["precisions"] = [best.precision]
    gemm_cfg["matrix_sizes"] = [best.matrix_size]
    gemm_cfg["impl"] = ["cuda"]

    save_yaml(cfg_path, cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Autotune selected GEMM kernel hyperparameters")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--bench-bin", default="build/lib/benchmark/bench_gemm", help="Path to bench_gemm binary")
    parser.add_argument("--versions", default="15,18,19,20", help="Comma-separated CUDA kernel versions")
    parser.add_argument("--modes", default="RC", help="Comma-separated op modes")
    parser.add_argument("--precisions", default="float16", help="Comma-separated precisions")
    parser.add_argument("--sizes", default="2048,4096", help="Comma-separated matrix sizes")
    parser.add_argument("--v20-modes", default="heuristic,dp,persistent", help="Comma-separated v20 decomposition modes")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs per trial")
    parser.add_argument("--measurement", type=int, default=40, help="Measurement runs per trial")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results in report")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout seconds per trial")
    parser.add_argument("--apply-best", action="store_true", help="Apply best result back to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print search space and exit")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = (repo_root / args.config).resolve()
    bench_bin = (repo_root / args.bench_bin).resolve()

    if not cfg_path.exists():
        print(f"ERROR: config file not found: {cfg_path}", file=sys.stderr)
        return 2
    if not bench_bin.exists():
        print(f"ERROR: bench binary not found: {bench_bin}", file=sys.stderr)
        print("Hint: run `make build` first.", file=sys.stderr)
        return 2

    versions = parse_list_int(args.versions)
    modes = parse_list_str(args.modes)
    precisions = parse_list_str(args.precisions)
    sizes = parse_list_int(args.sizes)
    v20_modes = parse_list_str(args.v20_modes)

    if not versions or not modes or not precisions or not sizes:
        print("ERROR: empty search space.", file=sys.stderr)
        return 2

    base_cfg = load_yaml(cfg_path)
    original_cfg = json.loads(json.dumps(base_cfg))

    grid = []
    for version, mode, precision, size in itertools.product(versions, modes, precisions, sizes):
        if version == 20:
            for vm in v20_modes:
                grid.append((version, mode, precision, size, vm))
        else:
            grid.append((version, mode, precision, size, None))

    print(f"[autotune] total trials: {len(grid)}")
    if args.dry_run:
        for i, g in enumerate(grid, start=1):
            print(f"  {i:03d}: version={g[0]} mode={g[1]} precision={g[2]} size={g[3]} v20_mode={g[4]}")
        return 0

    trials: List[Trial] = []
    t0 = time.time()

    try:
        for idx, (version, mode, precision, size, v20_mode) in enumerate(grid, start=1):
            print(
                f"[autotune] ({idx}/{len(grid)}) "
                f"v={version} mode={mode} precision={precision} size={size} v20_mode={v20_mode or '-'}"
            )
            trial = run_trial(
                repo_root=repo_root,
                bench_bin=bench_bin,
                cfg_path=cfg_path,
                base_cfg=base_cfg,
                version=version,
                mode=mode,
                precision=precision,
                matrix_size=size,
                warmup_runs=args.warmup,
                measurement_runs=args.measurement,
                v20_mode=v20_mode,
                timeout_s=args.timeout,
            )
            trials.append(trial)
            print(
                f"          => TFLOPS={trial.tflops:.3f}, speedup={trial.speedup:.3f}, latency={trial.latency_ms:.4f}ms"
            )
    finally:
        save_yaml(cfg_path, original_cfg)

    if not trials:
        print("ERROR: no successful trials.", file=sys.stderr)
        return 3

    ranked = sorted(trials, key=trial_sort_key, reverse=True)
    best = ranked[0]
    report_paths = write_reports(repo_root, trials, args.top_k)

    print("\n[autotune] done")
    print(f"[autotune] elapsed: {time.time() - t0:.1f}s")
    print(f"[autotune] best: version={best.version}, v20_mode={best.v20_mode or '-'}, mode={best.mode}, "
          f"precision={best.precision}, size={best.matrix_size}, TFLOPS={best.tflops:.3f}, speedup={best.speedup:.3f}")
    print(f"[autotune] report json: {report_paths['json']}")
    print(f"[autotune] report md:   {report_paths['md']}")

    if args.apply_best:
        apply_best_to_config(cfg_path, base_cfg, best)
        print(f"[autotune] applied best config to: {cfg_path}")
        if best.version == 20:
            print(f"[autotune] note: export KEBAB_V20_MODE={best.v20_mode or 'heuristic'} before benchmark/run")
    else:
        print("[autotune] config.yaml restored to original")
        if best.version == 20:
            print(f"[autotune] recommended env: export KEBAB_V20_MODE={best.v20_mode or 'heuristic'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
