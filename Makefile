PYTHON ?= python

.PHONY: setup audit test lint format \
	demo-backend demo-frontend demo-test demo-check \
	experiment-check experiment-validate \
	train-dry-run-yolov8n train-dry-run-yolov9t train-dry-run-yolo26n \
	smoke-suite-dry-run smoke-suite \
	full-suite-dry-run full-suite \
	evaluate-dry-run benchmark-dry-run validation-benchmark-dry-run validation-benchmark-suite gpu-preflight transfer-manifest transfer-verify \
	dataset-info dataset-download dataset-verify dataset-extract \
	dataset-inspect dataset-convert dataset-split dataset-validate \
	dataset-figures dataset-smoke dataset-prepare \
	calibrate calibrate-dry-run

setup:
	uv sync

audit:
	uv run python scripts/audit_environment.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

demo-backend:
	uv run uvicorn demo.backend.app.main:app --host 0.0.0.0 --port 8000 --reload

demo-frontend:
	cd demo/frontend && npm install && npm run dev

demo-test:
	uv run pytest -q

demo-check:
	uv run ruff check . && uv run ruff format --check . && cd demo/frontend && npm install && npm run lint && npm run typecheck && npm run build

experiment-check:
	uv run python scripts/train.py --config configs/experiment.yaml --dry-run

experiment-validate:
	uv run python scripts/train.py --config configs/experiment.yaml --preflight

train-dry-run-yolov8n:
	uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolov8.yaml --hardware-config configs/hardware/cpu-dev.yaml --run-config configs/runs/smoke.yaml --dry-run --smoke

train-dry-run-yolov9t:
	uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolov9.yaml --hardware-config configs/hardware/cpu-dev.yaml --run-config configs/runs/smoke.yaml --dry-run --smoke

train-dry-run-yolo26n:
	uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolo26.yaml --hardware-config configs/hardware/cpu-dev.yaml --run-config configs/runs/smoke.yaml --dry-run --smoke

train-dry-run:
	uv run python scripts/train.py --config configs/experiment.yaml --dry-run

smoke-suite-dry-run:
	uv run python scripts/run_smoke_suite.py --dry-run

smoke-suite:
	uv run python scripts/run_smoke_suite.py --execute

full-suite-dry-run:
	uv run python scripts/run_full_experiment_suite.py --dry-run $(if $(SUITE_ID),--suite-id $(SUITE_ID),) $(if $(MODELS),--models $(MODELS),) $(if $(RESUME),--resume,) $(if $(SKIP_COMPLETED),--skip-completed,)

full-suite:
	uv run python scripts/run_full_experiment_suite.py --execute $(if $(SUITE_ID),--suite-id $(SUITE_ID),) $(if $(MODELS),--models $(MODELS),) $(if $(RESUME),--resume,) $(if $(SKIP_COMPLETED),--skip-completed,) $(if $(RECOVER),--recover,) $(if $(CONTINUE_ON_ERROR),--continue-on-error,) $(if $(FORCE),--force,)

evaluate-dry-run:
	uv run python scripts/evaluate.py --config configs/experiment.yaml --checkpoint data/processed/yolo/dataset.yaml --dry-run

benchmark-dry-run:
	uv run python scripts/benchmark.py --config configs/experiment.yaml --checkpoint data/processed/yolo/dataset.yaml --dry-run

validation-benchmark-dry-run:
	uv run python scripts/run_validation_benchmark_suite.py --source-suite $(SOURCE_SUITE) --dry-run

validation-benchmark-suite:
	uv run python scripts/run_validation_benchmark_suite.py --source-suite $(SOURCE_SUITE) --execute

gpu-preflight:
	uv run python scripts/gpu_preflight.py

transfer-manifest:
	uv run python scripts/transfer_manifest.py

transfer-verify:
	uv run python scripts/transfer_manifest.py --verify

dataset-info:
	uv run python scripts/download_dataset.py --metadata-only

dataset-download:
	uv run python scripts/prepare_dataset.py download

dataset-verify:
	uv run python scripts/download_dataset.py --metadata-only

dataset-extract:
	uv run python scripts/prepare_dataset.py extract

dataset-inspect:
	uv run python scripts/prepare_dataset.py inspect --progress

dataset-convert:
	uv run python scripts/prepare_dataset.py convert --workers 8 --io-workers 16 --batch-size 32 --progress

dataset-split:
	uv run python scripts/prepare_dataset.py split

dataset-validate:
	uv run python scripts/prepare_dataset.py validate --io-workers 16 --batch-size 64 --progress

dataset-figures:
	uv run python scripts/prepare_dataset.py figures --progress

dataset-smoke:
	uv run python scripts/prepare_dataset.py smoke --max-batches 2

dataset-prepare:
	uv run python scripts/prepare_dataset.py prepare

calibrate:
	uv run python scripts/calibrate_rtx4090.py --execute

calibrate-dry-run:
	uv run python scripts/calibrate_rtx4090.py --dry-run
