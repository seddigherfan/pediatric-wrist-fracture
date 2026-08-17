# Decisions Log

## 2026-07-25

- Adopted Python 3.11 as the preferred environment target.
- Chose `uv` for dependency management and reproducible environments.
- Kept YAML-based layered configuration rather than introducing a heavier config framework.
- Deferred all real model execution until explicit `--execute` is supplied.
- Preserved patient-level immutable splits as the only dataset partitioning scheme.
- Kept the YOLOv9 checkpoint family limited to the verified upstream-compatible choice in the registry.
- Chose to record provenance atomically before any future execution starts.
- Chose not to archive or copy the completed dataset during transfer manifest generation.

## 2026-07-26

- Added separate hardware and run config layers to support CPU development and RTX 4090 execution profiles.
- Added run-directory markers for started, interrupted, and completed states.
- Added transfer-manifest verification so the GPU machine can confirm repository and dataset readiness before running experiments.
- Added a dedicated validation and benchmark suite that reuses the smoke-suite checkpoints, validates only `val`, and keeps smoke metrics marked as non-scientific pipeline verification outputs.
- Added a frozen Phase 5 full-training suite that executes YOLOv8n, YOLOv9t, and YOLO26n sequentially on one RTX 4090, requires validated calibration evidence, and refuses to use the test split.
