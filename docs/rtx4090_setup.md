# RTX 4090 Setup

These steps are for the GPU machine only.

## 1. Clone or copy the repository

```powershell
git clone <repository-url> pediatric-wrist-fracture
cd pediatric-wrist-fracture
```

If you are copying a prepared workspace instead of cloning, place it in a clean directory and open that directory in PowerShell.

## 2. Check out the intended commit

```powershell
git checkout <intended-commit>
```

## 3. Install `uv`

Follow the official `uv` installation instructions for Windows, then verify it:

```powershell
uv --version
```

## 4. Create a clean environment

```powershell
uv sync
```

If you want an isolated environment for the GPU machine, create it from the same repository state before any training work begins.

## 5. Install base dependencies

```powershell
uv sync --all-extras
```

## 6. Check `nvidia-smi`

```powershell
nvidia-smi
```

Confirm the driver version, RTX 4090 name, and reported VRAM.

## 7. Install the current official CUDA-enabled PyTorch build

Use the official PyTorch install selector on the GPU machine only:

https://pytorch.org/get-started/locally/

As of July 26, 2026, the official selector shows the current stable Windows CUDA wheel family as `cu126`, but you must re-check the selector on the GPU machine before installing.

Example pattern:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Do not treat that wheel family as permanent truth. Always confirm the current selector output first.

## 8. Verify `torch.cuda.is_available()`

```powershell
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 9. Verify GPU model and VRAM

```powershell
uv run python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name); print(round(p.total_memory/1024**3, 2))"
```

## 10. Verify the transfer manifest

```powershell
uv run python scripts/transfer_manifest.py --verify
```

## 11. Run GPU preflight

```powershell
uv run python scripts/gpu_preflight.py --require-gpu
```

## 12. Run train dry-run

```powershell
uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolo26.yaml --hardware-config configs/hardware/rtx4090.yaml --run-config configs/runs/full.yaml --dry-run
```

## 13. Run bounded smoke training

```powershell
uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolo26.yaml --hardware-config configs/hardware/rtx4090.yaml --run-config configs/runs/smoke.yaml --smoke --execute
```

This repository phase still keeps the actual training path guarded, so the smoke command is documented for the GPU machine workflow after Phase 3 approval.

## 14. Run the complete smoke suite

Dry run:

```powershell
uv run python scripts/run_smoke_suite.py --dry-run
```

Execute:

```powershell
uv run python scripts/run_smoke_suite.py --execute
```

The suite writes outputs under `outputs/smoke_suites/<suite_id>/`, reuses the existing training and recovery code, and refuses to run if smoke safety caps are violated.

## 15. Run validation and benchmarking on the smoke checkpoints

Dry run:

```powershell
uv run python scripts/run_validation_benchmark_suite.py --source-suite outputs/smoke_suites/<suite_id> --dry-run
```

Execute:

```powershell
uv run python scripts/run_validation_benchmark_suite.py --source-suite outputs/smoke_suites/<suite_id> --execute
```

This suite evaluates only the `val` split, benchmarks the completed smoke checkpoints sequentially on the RTX 4090, and keeps the resulting metrics explicitly labeled as pipeline-validation outputs rather than final thesis results.

## 16. Inspect artifacts

Check:

- `outputs/experiments/<model_family>/<run_id>/resolved_config.yaml`
- `outputs/experiments/<model_family>/<run_id>/provenance.json`
- `outputs/experiments/<model_family>/<run_id>/environment.json`
- `outputs/experiments/<model_family>/<run_id>/metrics/`
- `outputs/experiments/<model_family>/<run_id>/checkpoints/`

## 17. Run full experiments only after explicit approval

YOLO26 full run:

```powershell
uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolo26.yaml --hardware-config configs/hardware/rtx4090.yaml --run-config configs/runs/full.yaml --execute
```

YOLOv8n full run:

```powershell
uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolov8.yaml --hardware-config configs/hardware/rtx4090.yaml --run-config configs/runs/full.yaml --execute
```

YOLOv9t full run:

```powershell
uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolov9.yaml --hardware-config configs/hardware/rtx4090.yaml --run-config configs/runs/full.yaml --execute
```

## 18. Run the Phase 5 full experiment suite

Dry run:

```powershell
uv run python scripts/run_full_experiment_suite.py --dry-run
```

Execute:

```powershell
uv run python scripts/run_full_experiment_suite.py --execute
```

Resume an existing suite:

```powershell
uv run python scripts/run_full_experiment_suite.py --suite-id <existing-suite-id> --resume --execute
```

Reuse completed model runs when compatible:

```powershell
uv run python scripts/run_full_experiment_suite.py --suite-id <existing-suite-id> --skip-completed --execute
```

The suite trains YOLOv8n, YOLOv9t, and YOLO26n sequentially on the single RTX 4090, records only training and validation metrics, and never uses the test split during Phase 5.
