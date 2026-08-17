# Experiment Protocol

## Objective

Compare YOLOv8n, YOLOv9t, and YOLO26n on the same immutable pediatric wrist fracture dataset, using the same patient-level splits, image size, seed, and selection metric.

## Shared controls

- Dataset: processed YOLO dataset derived from GRAZPEDWRI-DX
- Splits: immutable patient-level train/val/test splits
- Image size: 640 unless a smoke config explicitly reduces it for CPU-safe validation
- Seed: 42
- Maximum epochs: set by the selected run config
- Patience: set by the selected experiment config
- Pretrained initialization: enabled for all primary comparisons
- Augmentation policy: conservative and fixed per experiment config
- AMP policy: GPU-run default enabled, CPU-run disabled
- Test isolation: test split is held out from model selection
- Model-selection metric: `metrics/mAP50-95(B)`
- Latency benchmark protocol: fixed batch size 1, bounded warmup, bounded measured samples, recorded environment and checkpoint metadata
- Validation policy: val split only for this phase; test split is held out until explicitly approved later
- Phase 5 full experiments run the frozen protocol sequentially on a single RTX 4090 using `uv run python scripts/run_full_experiment_suite.py --execute`
- Phase 5 dry-runs use `uv run python scripts/run_full_experiment_suite.py --dry-run`
- Phase 5 requires validated calibration evidence and `configs/runs/full.yaml` to remain frozen at the applied calibrated batch
- The test split is never used during Phase 5 training or model selection
- Recovery may reuse valid raw artifacts without retraining when postprocessing fails
- Skip-completed mode reuses only fully validated completed runs

## Primary models

### YOLOv8n

- Checkpoint: `yolov8n.pt`
- Parameters: verified by upstream Ultralytics documentation at run time
- FLOPs: pending GPU-machine verification if needed for the final comparison table
- Checkpoint size: pending GPU-machine verification
- Training support: supported through the Ultralytics API
- Inference mode: standard Ultralytics detection pipeline
- NMS / end-to-end behavior: standard detection model with NMS in default inference mode
- Source / license: Ultralytics model family documentation and repository license

### YOLOv9t

- Checkpoint: `yolov9t.pt`
- Parameters: pending GPU-machine verification
- FLOPs: pending GPU-machine verification
- Checkpoint size: pending GPU-machine verification
- Training support: supported through the configured Ultralytics-compatible pipeline in this repository
- Inference mode: standard detection pipeline
- NMS / end-to-end behavior: standard detection model in default inference mode
- Source / license: official Ultralytics-compatible documentation and repository license

### YOLO26n

- Checkpoint: `yolo26n.pt`
- Parameters: pending GPU-machine verification
- FLOPs: pending GPU-machine verification
- Checkpoint size: pending GPU-machine verification
- Training support: supported through the Ultralytics API
- Inference mode: standard detection pipeline
- NMS / end-to-end behavior: default behavior is NMS-free-friendly in the registry metadata, but the actual benchmark path still records the resolved execution mode
- Source / license: Ultralytics YOLO26 documentation and repository license

## Execution rules

- `--dry-run` validates config composition and prints the resolved plan.
- `--preflight` performs the same validation path without execution.
- `--execute` is required before any real training, evaluation, or benchmarking path may run.
- CPU full training is rejected.
- Test-set evaluation requires `--allow-test`.
- Evaluation and benchmarking require an explicit checkpoint path.
- Validation and benchmark runs reuse the completed smoke checkpoints and do not alter dataset manifests or splits.
- The unified suite command is `uv run python scripts/run_validation_benchmark_suite.py --source-suite <path> --execute`.
- The Phase 5 full-training suite command is `uv run python scripts/run_full_experiment_suite.py --execute`.
- The Phase 3 smoke suite is run through `uv run python scripts/run_smoke_suite.py --dry-run` or `uv run python scripts/run_smoke_suite.py --execute`.
- Smoke-suite outputs are stored under `outputs/smoke_suites/<suite_id>/`.
- Smoke metrics are for pipeline verification only and are not scientific results.

## Outputs

Each run writes a resolved config, provenance, environment metadata, logs, checkpoints, raw outputs, metrics, figures, and benchmarks into `outputs/experiments/<model_family>/<run_id>/`.
Phase 5 suite artifacts are written under `outputs/full_experiment_suites/<suite_id>/`.
