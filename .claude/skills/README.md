# Kebab Copilot Skills

This directory contains project-scoped Agent Skills for GitHub Copilot.

Each skill is grounded in real commands and files in this repository:

- Build and environment setup via `Makefile` and `kebab/CMakeLists.txt`
- Benchmarks and run targets via `Makefile` and `kebab/lib/benchmark/CMakeLists.txt`
- Microbenchmarks, dump, and profiling workflows via `Makefile` and `scripts/`
- GEMM tuning and configuration via `config.yaml` and `docs/`

Use `scripts/validate_copilot_skills.sh` to run a minimal structural validation.
