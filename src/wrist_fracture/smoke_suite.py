from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrist_fracture.config import ConfigError, ExperimentConfig, load_config_bundle  # noqa: E402
from wrist_fracture.environment import collect_environment_metadata  # noqa: E402
from wrist_fracture.models.registry import resolve_model_spec  # noqa: E402
from wrist_fracture.provenance import (  # noqa: E402
    collect_environment_report,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    sha256_file,
    to_jsonable,
)
from wrist_fracture.runtime import (  # noqa: E402
    execute_training_with_args,
    now_utc,
    persist_run_metadata,
    recover_training_artifacts,
    run_root,
)
from wrist_fracture.transfer_manifest import build_manifest, verify_manifest  # noqa: E402

MODEL_ORDER = ("yolov8", "yolov9", "yolo26")
MODEL_CONFIGS = {
    "yolov8": Path("configs/models/yolov8.yaml"),
    "yolov9": Path("configs/models/yolov9.yaml"),
    "yolo26": Path("configs/models/yolo26.yaml"),
}
SMOKE_CAPS = {"epochs": 1, "image_size": 320, "batch_size": 4, "patience": 1, "repeated_runs": 1}
_now = now_utc


def default_suite_id() -> str:
    return f"smoke-{_now().replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:6]}"


def normalize_models(raw: str | None) -> list[str]:
    if not raw:
        return list(MODEL_ORDER)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(items) - set(MODEL_ORDER))
    if unknown:
        raise ConfigError(f"unknown smoke models: {', '.join(unknown)}")
    return [model for model in MODEL_ORDER if model in items]


def build_config(model_family: str, suite_id: str, *, root: Path) -> ExperimentConfig:
    cfg = load_config_bundle(
        root / "configs/experiment.yaml",
        model_path=MODEL_CONFIGS[model_family],
        hardware_path=root / "configs/hardware/rtx4090.yaml",
        run_path=root / "configs/runs/smoke.yaml",
    )
    return ExperimentConfig(
        dataset_yaml=cfg.dataset_yaml,
        dataset_split_yaml=cfg.dataset_split_yaml,
        model=cfg.model,
        hardware=cfg.hardware,
        run=type(cfg.run)(
            name=cfg.run.name,
            output_root=root / "outputs" / "experiments",
            run_id=f"{model_family}/{suite_id}",
            resume=False,
            save_period=cfg.run.save_period,
            validation_split="val",
            test_split="test",
            allow_test_evaluation=False,
            selection_metric=cfg.run.selection_metric,
            repeated_runs=1,
            batch_size_policy="fixed",
        ),
        image_size=min(cfg.image_size, SMOKE_CAPS["image_size"]),
        epochs=1,
        patience=1,
        seed=cfg.seed,
        pretrained=cfg.pretrained,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        augmentation=cfg.augmentation,
        resume_checkpoint=cfg.resume_checkpoint,
        save_json=cfg.save_json,
        batch_size=min(cfg.batch_size, SMOKE_CAPS["batch_size"]),
        extra=cfg.extra,
    )


def validate_smoke_caps(cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    if cfg.epochs != 1:
        errors.append("epochs must equal 1")
    if cfg.image_size > SMOKE_CAPS["image_size"]:
        errors.append("image size exceeds smoke cap")
    if cfg.batch_size > SMOKE_CAPS["batch_size"]:
        errors.append("batch size exceeds smoke cap")
    if cfg.patience > SMOKE_CAPS["patience"]:
        errors.append("patience exceeds smoke cap")
    if cfg.run.repeated_runs != 1:
        errors.append("repeated_runs must equal 1")
    if cfg.run.validation_split != "val":
        errors.append("validation split must be val")
    if cfg.run.allow_test_evaluation:
        errors.append("test evaluation must remain disabled")
    if cfg.run.test_split == "val":
        errors.append("test split must remain disabled")
    return errors


def _repo_preflight(root: Path) -> list[str]:
    missing = [
        path
        for path in [
            root / "configs/experiment.yaml",
            root / "configs/runs/smoke.yaml",
            root / "configs/hardware/rtx4090.yaml",
        ]
        if not path.exists()
    ]
    return [f"missing {path}" for path in missing]


def _gpu_preflight() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise ConfigError(f"CUDA preflight failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise ConfigError("CUDA must be available")
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise ConfigError("requested CUDA device does not exist")
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_count": device_count,
        "device_name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
    }


def _dataset_preflight(root: Path, dataset_yaml: Path) -> dict[str, Any]:
    manifest = build_manifest(root, dataset_yaml)
    errors = verify_manifest(manifest, root, dataset_yaml)
    if errors:
        raise ConfigError("; ".join(errors))
    return manifest


def _artifact_paths(root: Path) -> list[Path]:
    return [
        root / "resolved_config.yaml",
        root / "environment.json",
        root / "provenance.json",
        root / "raw/train/results.csv",
        root / "checkpoints/best.pt",
        root / "checkpoints/last.pt",
        root / "metrics/history.csv",
        root / "metrics/validation.json",
        root / "metrics/run_summary.json",
        root / "completed.marker",
    ]


def validate_artifacts(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _artifact_paths(root):
        if not path.exists():
            errors.append(f"missing artifact: {path.relative_to(root)}")
    if (root / "interrupted.marker").exists():
        errors.append("interrupted.marker must not exist")
    for rel in ["metrics/validation.json", "metrics/run_summary.json"]:
        path = root / rel
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                errors.append(f"invalid json: {rel}")
                continue
            if (
                not isinstance(payload, dict)
                or "schema_version" not in payload
                and rel.endswith("validation.json")
            ):
                errors.append(f"invalid schema: {rel}")
    return errors


def summarize_run(
    cfg: ExperimentConfig, root: Path, *, status: str, recovery_used: bool, error: str | None = None
) -> dict[str, Any]:
    validation = json.loads((root / "metrics/validation.json").read_text(encoding="utf-8"))
    run_summary = json.loads((root / "metrics/run_summary.json").read_text(encoding="utf-8"))
    checkpoint = root / "checkpoints/best.pt"
    spec = resolve_model_spec(cfg.model)
    size_mb = round(checkpoint.stat().st_size / 1024**2, 3) if checkpoint.exists() else None
    return {
        "model_family": cfg.model.family,
        "model_scale": cfg.model.scale,
        "status": status,
        "run_id": cfg.run.run_id,
        "run_path": str(root),
        "checkpoint": str(checkpoint),
        "start_time": run_summary.get("started_at"),
        "end_time": run_summary.get("ended_at"),
        "duration_seconds": run_summary.get("duration_seconds"),
        "gpu": run_summary.get("gpu_peak_memory_bytes"),
        "peak_vram_gb": round(run_summary["gpu_peak_memory_bytes"] / 1024**3, 3)
        if run_summary.get("gpu_peak_memory_bytes")
        else None,
        "params_m": spec.params_m,
        "flops_b": spec.flops_b,
        "checkpoint_size_mb": size_mb,
        "precision": validation.get("precision"),
        "recall": validation.get("recall"),
        "f1": validation.get("f1"),
        "map50": validation.get("map50"),
        "map50_95": validation.get("map50_95"),
        "best_epoch": validation.get("best_epoch"),
        "recovery_used": recovery_used,
        "error_summary": error,
        "dataset_yaml_sha256": sha256_file(cfg.dataset_yaml) if cfg.dataset_yaml.exists() else None,
        "train_split": "train",
        "validation_split": cfg.run.validation_split,
        "epochs": cfg.epochs,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "patience": cfg.patience,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "augmentation": cfg.augmentation,
        "workers": cfg.hardware.workers,
        "amp": cfg.hardware.amp,
        "cache": cfg.hardware.cache,
    }


def compare_configs(runs: list[dict[str, Any]]) -> list[str]:
    if len(runs) < 2:
        return []
    keys = [
        "dataset_yaml_sha256",
        "train_split",
        "validation_split",
        "epochs",
        "image_size",
        "batch_size",
        "seed",
        "patience",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "augmentation",
        "workers",
        "amp",
        "cache",
    ]
    diffs: list[str] = []
    for key in keys:
        values = {run["model_family"]: run.get(key) for run in runs}
        normalized = {
            model: json.dumps(value, sort_keys=True, default=str) for model, value in values.items()
        }
        if len(set(normalized.values())) > 1:
            diffs.append(f"{key}: {values}")
    return diffs


def build_reports(suite_dir: Path, summaries: list[dict[str, Any]]) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_summary = {"models": summaries, "generated_at": _now()}
    (suite_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = [
        {
            k: summary.get(k)
            for k in [
                "status",
                "run_id",
                "run_path",
                "checkpoint",
                "start_time",
                "end_time",
                "duration_seconds",
                "gpu",
                "peak_vram_gb",
                "params_m",
                "flops_b",
                "checkpoint_size_mb",
                "precision",
                "recall",
                "f1",
                "map50",
                "map50_95",
                "best_epoch",
                "recovery_used",
                "error_summary",
            ]
        }
        for summary in summaries
    ]
    with (suite_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    report = ["# Smoke Suite", "", "Smoke metrics are not scientific results."]
    for summary in summaries:
        report.extend(
            [
                "",
                f"## {summary['model_family']}",
                f"- status: {summary['status']}",
                f"- run_id: {summary['run_id']}",
                f"- run_path: {summary['run_path']}",
                f"- checkpoint: {summary['checkpoint']}",
                f"- Precision: {summary.get('precision')}",
                f"- Recall: {summary.get('recall')}",
                f"- F1: {summary.get('f1')}",
                f"- mAP@0.5: {summary.get('map50')}",
                f"- mAP@0.5:0.95: {summary.get('map50_95')}",
            ]
        )
    (suite_dir / "suite_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_environment_json(root: Path, suite_id: str) -> dict[str, Any]:
    return {
        "suite_id": suite_id,
        "timestamp_utc": _now(),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "dependency_lock_sha256": dependency_lock_hash(root),
        "environment": to_jsonable(collect_environment_report(root)),
        "runtime": to_jsonable(collect_environment_metadata(root)),
    }


@dataclass
class SuiteResult:
    suite_id: str
    suite_dir: Path
    summaries: list[dict[str, Any]]
    errors: list[str]


def run_suite(args: argparse.Namespace) -> int:
    root = Path.cwd()
    suite_id = args.suite_id or default_suite_id()
    models = normalize_models(args.models)
    suite_dir = root / "outputs" / "smoke_suites" / suite_id
    if suite_dir.exists() and not (args.recover or args.skip_completed):
        raise ConfigError(f"suite collision: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "running.marker").write_text(_now(), encoding="utf-8")
    log_path = suite_dir / "suite.log"
    commands: list[str] = []
    summaries: list[dict[str, Any]] = []

    def log(message: str) -> None:
        print(message)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    try:
        log("stage 1: repository and configuration preflight")
        repo_errors = _repo_preflight(root)
        if repo_errors:
            raise ConfigError("; ".join(repo_errors))
        log("stage 2: gpu preflight with GPU required")
        _gpu_preflight()
        log("stage 3: dataset and transfer-manifest verification")
        manifest = _dataset_preflight(root, root / "data/processed/yolo/dataset.yaml")
        log("stage 4: dry-run resolution for all models")
        for model in models:
            cfg = build_config(model, suite_id, root=root)
            caps = validate_smoke_caps(cfg)
            if caps:
                raise ConfigError("; ".join(caps))
            cmd = (
                "uv run python scripts/train.py --config configs/experiment.yaml "
                f"--model-config {MODEL_CONFIGS[model].as_posix()} "
                "--hardware-config configs/hardware/rtx4090.yaml "
                "--run-config configs/runs/smoke.yaml --dry-run --smoke"
            )
            commands.append(cmd)
            if args.print_commands:
                log(cmd)
        if args.dry_run and not args.execute:
            build_reports(suite_dir, [])
            (suite_dir / "environment.json").write_text(
                json.dumps(build_environment_json(root, suite_id), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (suite_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
            (suite_dir / "completed.marker").write_text(_now(), encoding="utf-8")
            return 0
        log("stage 5: sequential one-epoch smoke training")
        for model in models:
            cfg = build_config(model, suite_id, root=root)
            run_root_path = run_root(cfg, cfg.run.run_id or "")
            if args.skip_completed and (run_root_path / "completed.marker").exists():
                summaries.append(
                    {
                        **summarize_run(
                            cfg, run_root_path, status="skipped", recovery_used=False, error=None
                        ),
                        "status": "skipped",
                    }
                )
                continue
            if run_root_path.exists() and (run_root_path / "completed.marker").exists():
                raise ConfigError(f"completed run may not be overwritten: {run_root_path}")
            log(f"model {model}: train")
            train_args = argparse.Namespace(execute=True, smoke=True)
            persist_run_metadata(run_root_path, cfg, train_args)
            error: str | None = None
            recovery_used = False
            try:
                execute_training_with_args(cfg, run_root_path, train_args)
            except Exception as exc:
                error = str(exc)
                if (run_root_path / "raw").exists() and args.recover:
                    log(f"model {model}: recovery")
                    recover_training_artifacts(run_root_path)
                    recovery_used = True
                elif not args.continue_on_error:
                    raise
            if error is None and (run_root_path / "completed.marker").exists():
                validation_errors = validate_artifacts(run_root_path)
                if validation_errors:
                    error = "; ".join(validation_errors)
                    if (run_root_path / "raw").exists() and args.recover:
                        log(f"model {model}: recovery")
                        recover_training_artifacts(run_root_path)
                        recovery_used = True
                        validation_errors = validate_artifacts(run_root_path)
                        if validation_errors:
                            error = "; ".join(validation_errors)
                if error:
                    if not args.continue_on_error:
                        raise ConfigError(error)
                    summaries.append(
                        {
                            **summarize_run(
                                cfg,
                                run_root_path,
                                status="failed",
                                recovery_used=recovery_used,
                                error=error,
                            ),
                            "status": "failed",
                        }
                    )
                    continue
                summaries.append(
                    summarize_run(
                        cfg, run_root_path, status="completed", recovery_used=recovery_used
                    )
                )
            else:
                if error:
                    if not args.continue_on_error:
                        raise ConfigError(error)
                    summaries.append(
                        {
                            **summarize_run(
                                cfg,
                                run_root_path,
                                status="failed",
                                recovery_used=recovery_used,
                                error=error,
                            ),
                            "status": "failed",
                        }
                    )
        log("stage 6: post-training artifact validation")
        for summary in summaries:
            if summary["status"] == "completed":
                errors = validate_artifacts(Path(summary["run_path"]))
                if errors:
                    raise ConfigError("; ".join(errors))
        log("stage 7: automatic postprocessing recovery")
        log("stage 8: cross-model smoke consistency audit")
        diffs = compare_configs(
            [summary for summary in summaries if summary["status"] == "completed"]
        )
        log("stage 9: unified summary generation")
        build_reports(suite_dir, summaries)
        (suite_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
        (suite_dir / "environment.json").write_text(
            json.dumps(
                {
                    **build_environment_json(root, suite_id),
                    "transfer_manifest": manifest,
                    "consistency_differences": diffs,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if any(summary["status"] != "completed" for summary in summaries):
            (suite_dir / "failed.marker").write_text(_now(), encoding="utf-8")
            raise ConfigError("one or more smoke models failed")
        (suite_dir / "completed.marker").write_text(_now(), encoding="utf-8")
        return 0
    except Exception:
        (suite_dir / "failed.marker").write_text(_now(), encoding="utf-8")
        if (suite_dir / "completed.marker").exists():
            (suite_dir / "completed.marker").unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--suite-id")
    parser.add_argument("--models")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    if args.dry_run and not args.execute:
        plan = {
            "suite_id": args.suite_id or default_suite_id(),
            "models": normalize_models(args.models),
            "caps": SMOKE_CAPS,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
