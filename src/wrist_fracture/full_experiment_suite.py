from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from wrist_fracture.calibration import (
    discover_latest_completed_execution,
    load_calibration_report,
    report_has_complete_real_evidence,
)
from wrist_fracture.config import ConfigError, ExperimentConfig, load_config_bundle
from wrist_fracture.models.registry import resolve_model_spec
from wrist_fracture.provenance import (
    collect_environment_report,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    sha256_file,
    to_jsonable,
)
from wrist_fracture.runtime import (
    execute_training_with_args,
    now_utc,
    persist_run_metadata,
    recover_training_artifacts,
    run_root,
)
from wrist_fracture.transfer_manifest import build_manifest, verify_manifest

MODEL_ORDER = ("yolov8", "yolov9", "yolo26")
MODEL_CONFIGS = {
    "yolov8": Path("configs/models/yolov8.yaml"),
    "yolov9": Path("configs/models/yolov9.yaml"),
    "yolo26": Path("configs/models/yolo26.yaml"),
}
FULL_RUN_CONFIG = Path("configs/runs/full.yaml")
OUTPUT_ROOT = Path("outputs/full_experiment_suites")
_EXPECTED_SHARED_KEYS = [
    "dataset_yaml_sha256",
    "train_split",
    "validation_split",
    "image_size",
    "batch_size",
    "epochs",
    "patience",
    "seed",
    "optimizer",
    "lr0",
    "lrf",
    "weight_decay",
    "augmentation",
    "workers",
    "cache",
    "amp",
    "deterministic",
    "save_period",
    "selection_metric",
]
_EXPECTED_EFFECTIVE_KEYS = [
    "optimizer",
    "lr0",
    "momentum",
    "augmentation",
]


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if isinstance(payload, str):
        tmp.write_text(payload, encoding="utf-8")
    else:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_marker(path: Path) -> None:
    _atomic_write(path, now_utc())


def default_suite_id() -> str:
    return f"full-{now_utc().replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:6]}"


def parse_models(raw: str | None) -> list[str]:
    if not raw:
        return list(MODEL_ORDER)
    models = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ConfigError(f"unknown full-suite models: {', '.join(unknown)}")
    return [model for model in MODEL_ORDER if model in models]


def _repo_root() -> Path:
    return Path.cwd()


def _load_model_config(model_family: str) -> ExperimentConfig:
    root = _repo_root()
    return load_config_bundle(
        root / "configs/experiment.yaml",
        model_path=root / MODEL_CONFIGS[model_family],
        hardware_path=root / "configs/hardware/rtx4090.yaml",
        run_path=root / FULL_RUN_CONFIG,
    )


def build_full_config(model_family: str, suite_id: str) -> ExperimentConfig:
    cfg = _load_model_config(model_family)
    return ExperimentConfig(
        dataset_yaml=cfg.dataset_yaml,
        dataset_split_yaml=cfg.dataset_split_yaml,
        model=cfg.model,
        hardware=cfg.hardware,
        run=type(cfg.run)(
            name=cfg.run.name,
            output_root=cfg.run.output_root,
            run_id=f"{suite_id}-{model_family}",
            resume=cfg.run.resume,
            save_period=cfg.run.save_period,
            validation_split=cfg.run.validation_split,
            test_split=cfg.run.test_split,
            allow_test_evaluation=cfg.run.allow_test_evaluation,
            selection_metric=cfg.run.selection_metric,
            repeated_runs=cfg.run.repeated_runs,
            batch_size_policy=cfg.run.batch_size_policy,
        ),
        image_size=cfg.image_size,
        epochs=cfg.epochs,
        patience=cfg.patience,
        seed=cfg.seed,
        pretrained=cfg.pretrained,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        augmentation=cfg.augmentation,
        resume_checkpoint=cfg.resume_checkpoint,
        save_json=cfg.save_json,
        batch_size=cfg.batch_size,
        extra=cfg.extra,
    )


def build_run_path(cfg: ExperimentConfig) -> Path:
    return run_root(cfg, cfg.run.run_id or "")


def _cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_device_exists(device: str) -> bool:
    if not device.startswith("cuda"):
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        index = 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        return index < torch.cuda.device_count()
    except Exception:
        return False


def _dataset_preflight(root: Path, dataset_yaml: Path) -> dict[str, Any]:
    manifest = build_manifest(root, dataset_yaml)
    errors = verify_manifest(manifest, root, dataset_yaml)
    if errors:
        raise ConfigError("; ".join(errors))
    return manifest


def _calibration_root(root: Path) -> Path:
    return root / "outputs" / "calibration"


def _calibration_application_record(root: Path) -> dict[str, Any] | None:
    calibration_root = _calibration_root(root)
    latest = discover_latest_completed_execution(calibration_root)
    if latest is None:
        return None
    report = load_calibration_report(latest / "calibration_report.json") or {}
    if not report_has_complete_real_evidence(report):
        return None
    record = load_calibration_report(latest / "application_record.json")
    report["calibration_root"] = str(latest)
    report["application_record"] = record
    return report


def validate_calibration_evidence(root: Path, cfg: ExperimentConfig) -> dict[str, Any]:
    report = _calibration_application_record(root)
    if not report:
        raise ConfigError("calibration evidence is missing or incomplete")
    selected = report.get("common_stable_batch")
    if selected is None:
        raise ConfigError("calibration evidence does not define a common stable batch")
    if int(selected) != 64 and int(selected) != int(cfg.batch_size):
        raise ConfigError("calibration batch is incompatible with the frozen full protocol")
    application = report.get("application_record") or {}
    if application:
        if application.get("selected_batch") != selected:
            raise ConfigError("calibration application record does not match the evidence")
        if application.get("applied_batch") != cfg.batch_size:
            raise ConfigError("applied calibration batch did not propagate into full config")
    return report


def validate_full_protocol_configs(configs: list[ExperimentConfig]) -> list[str]:
    if not configs:
        return ["no models selected"]
    base = _run_signature(configs[0])
    diffs: list[str] = []
    for cfg in configs[1:]:
        other = _run_signature(cfg)
        for key in _EXPECTED_SHARED_KEYS:
            if json.dumps(base.get(key), sort_keys=True, default=str) != json.dumps(
                other.get(key), sort_keys=True, default=str
            ):
                diffs.append(f"{cfg.model.family}.{key}: {base.get(key)!r} != {other.get(key)!r}")
    return diffs


def _run_signature(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset_yaml_sha256": sha256_file(cfg.dataset_yaml) if cfg.dataset_yaml.exists() else None,
        "train_split": "train",
        "validation_split": cfg.run.validation_split,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "augmentation": cfg.augmentation,
        "workers": cfg.hardware.workers,
        "cache": cfg.hardware.cache,
        "amp": cfg.hardware.amp,
        "deterministic": cfg.hardware.deterministic,
        "save_period": cfg.run.save_period,
        "selection_metric": cfg.run.selection_metric,
        "allow_test_evaluation": cfg.run.allow_test_evaluation,
        "test_split": cfg.run.test_split,
    }


def validate_full_config(cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    if cfg.epochs <= 1:
        errors.append("epochs must be greater than 1 for the full suite")
    if cfg.image_size != 640:
        errors.append("image_size must equal 640 for the frozen full protocol")
    if cfg.run.allow_test_evaluation:
        errors.append("test evaluation must be disabled")
    if cfg.run.test_split == "test" and cfg.run.allow_test_evaluation:
        errors.append("test split must remain disabled")
    if cfg.batch_size != 64:
        errors.append("batch size must equal the calibrated full value 64")
    return errors


def validate_dataset_and_splits(root: Path, cfg: ExperimentConfig) -> dict[str, Any]:
    manifest = _dataset_preflight(root, cfg.dataset_yaml)
    split_audit = root / "outputs/dataset_reports/final_dataset_audit.json"
    if not split_audit.exists():
        raise ConfigError("final dataset audit is missing")
    return manifest


def validate_completed_model_run(root: Path, cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    required = [
        "resolved_config.yaml",
        "environment.json",
        "provenance.json",
        "command.txt",
        "raw",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "metrics/history.csv",
        "metrics/validation.json",
        "metrics/run_summary.json",
        "completed.marker",
    ]
    for rel in required:
        path = root / rel
        if not path.exists():
            errors.append(f"missing artifact: {rel}")
    if (root / "interrupted.marker").exists():
        errors.append("interrupted.marker must not exist")
    return errors


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def summarize_model_run(
    cfg: ExperimentConfig, root: Path, *, status: str, recovery_used: bool, error: str | None = None
) -> dict[str, Any]:
    validation = _read_json_or_empty(root / "metrics/validation.json")
    run_summary = _read_json_or_empty(root / "metrics/run_summary.json")
    spec = resolve_model_spec(cfg.model)
    checkpoint = root / "checkpoints/best.pt"
    return {
        "model_family": cfg.model.family,
        "checkpoint": spec.checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
        "run_id": cfg.run.run_id,
        "run_path": str(root),
        "status": status,
        "start_time": run_summary.get("started_at"),
        "end_time": run_summary.get("ended_at"),
        "total_training_duration_seconds": run_summary.get("duration_seconds"),
        "completed_epochs": run_summary.get("completed_epochs"),
        "best_epoch": validation.get("best_epoch"),
        "early_stopping_status": run_summary.get("early_stopping_status"),
        "early_stopping_reason": run_summary.get("early_stopping_reason"),
        "final_precision": validation.get("precision"),
        "final_recall": validation.get("recall"),
        "final_f1": validation.get("f1"),
        "final_map50": validation.get("map50"),
        "final_map50_95": validation.get("map50_95"),
        "best_validation_map50_95": validation.get("map50_95"),
        "peak_allocated_vram_bytes": run_summary.get("gpu_peak_memory_bytes"),
        "peak_reserved_vram_bytes": run_summary.get("gpu_peak_memory_bytes"),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "params_m": spec.params_m,
        "flops_b": spec.flops_b,
        "resume_count": run_summary.get("resume_count", 0),
        "recovery_used": recovery_used,
        "error_summary": error,
        "effective_protocol": run_summary.get("effective_protocol"),
        **_run_signature(cfg),
    }


def _suite_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "model_family",
        "checkpoint",
        "checkpoint_sha256",
        "run_id",
        "run_path",
        "status",
        "start_time",
        "end_time",
        "total_training_duration_seconds",
        "completed_epochs",
        "best_epoch",
        "early_stopping_status",
        "early_stopping_reason",
        "final_precision",
        "final_recall",
        "final_f1",
        "final_map50",
        "final_map50_95",
        "best_validation_map50_95",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
        "checkpoint_size_bytes",
        "params_m",
        "flops_b",
        "resume_count",
        "recovery_used",
        "error_summary",
        "effective_protocol",
    ]
    return [{key: summary.get(key) for key in keys} for summary in summaries]


def _effective_protocol_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    protocol = summary.get("effective_protocol")
    if not isinstance(protocol, dict):
        return None
    return {
        "optimizer": protocol.get("optimizer"),
        "lr0": protocol.get("lr0"),
        "momentum": protocol.get("momentum"),
        "augmentation": protocol.get("augmentation"),
    }


def _validate_effective_protocols(summaries: list[dict[str, Any]]) -> list[str]:
    if not summaries:
        return []
    protocols = [_effective_protocol_summary(summary) for summary in summaries]
    present = [protocol for protocol in protocols if protocol is not None]
    if not present:
        return []
    base = present[0]
    diffs: list[str] = []
    for summary, other in zip(summaries[1:], protocols[1:], strict=False):
        if other is None:
            diffs.append(f"{summary.get('model_family')}: missing effective protocol")
            continue
        for key in _EXPECTED_EFFECTIVE_KEYS:
            if json.dumps(base.get(key), sort_keys=True, default=str) != json.dumps(
                other.get(key), sort_keys=True, default=str
            ):
                diffs.append(
                    f"{summary.get('model_family')}.{key}: {base.get(key)!r} != {other.get(key)!r}"
                )
    return diffs


def _write_suite_report(
    suite_dir: Path, summaries: list[dict[str, Any]], differences: list[str]
) -> None:
    suite_dir.mkdir(parents=True, exist_ok=True)
    payload = {"models": summaries, "generated_at": now_utc(), "differences": differences}
    _atomic_write(suite_dir / "suite_summary.json", payload)
    rows = _suite_summary_rows(summaries)
    with (suite_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    report = [
        "# Full Experiment Suite",
        "",
        "The test split was not used during full training or model selection.",
    ]
    if differences:
        report.extend(["", "## Consistency Differences", *[f"- {item}" for item in differences]])
    for summary in summaries:
        report.extend(
            [
                "",
                f"## {summary['model_family']}",
                f"- status: {summary['status']}",
                f"- run_id: {summary['run_id']}",
                f"- run_path: {summary['run_path']}",
                f"- checkpoint: {summary['checkpoint']}",
                f"- best_validation_map50_95: {summary.get('best_validation_map50_95')}",
                f"- final_map50_95: {summary.get('final_map50_95')}",
            ]
        )
    _atomic_write(suite_dir / "suite_report.md", "\n".join(report) + "\n")


def _write_failure_marker(
    path: Path, *, reason: str, summary: dict[str, Any] | None = None
) -> None:
    _atomic_write(
        path,
        {
            "timestamp_utc": now_utc(),
            "reason": reason,
            "summary": summary or {},
        },
    )


def _planned_command(model_family: str) -> str:
    return (
        "uv run python scripts/train.py --config configs/experiment.yaml "
        f"--model-config {MODEL_CONFIGS[model_family].as_posix()} "
        "--hardware-config configs/hardware/rtx4090.yaml "
        f"--run-config {FULL_RUN_CONFIG.as_posix()} --execute"
    )


def _suite_paths(suite_id: str) -> Path:
    return OUTPUT_ROOT / suite_id


def _safe_gpu_preflight(cfg: ExperimentConfig) -> None:
    if not _cuda_is_available():
        raise ConfigError("CUDA must be available for full training")
    if not _cuda_device_exists(cfg.hardware.device):
        raise ConfigError("requested CUDA device does not exist")
    if cfg.hardware.device == "cpu":
        raise ConfigError("CPU full training is disabled")


def _run_path_for(cfg: ExperimentConfig) -> Path:
    return build_run_path(cfg)


def _collision_allowed(path: Path, *, resume: bool, skip_completed: bool, force: bool) -> bool:
    if not path.exists():
        return True
    if resume or skip_completed or force:
        return True
    return False


@dataclass
class SuiteResult:
    suite_id: str
    suite_dir: Path
    summaries: list[dict[str, Any]]
    differences: list[str]


def run_suite(args: argparse.Namespace) -> int:
    root = _repo_root()
    suite_id = args.suite_id or default_suite_id()
    suite_dir = _suite_paths(suite_id)
    models = parse_models(args.models)
    configs = [build_full_config(model, suite_id) for model in models]
    if suite_dir.exists() and not _collision_allowed(
        suite_dir, resume=args.resume, skip_completed=args.skip_completed, force=args.force
    ):
        raise ConfigError(f"output collision: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    _write_marker(suite_dir / "planned.marker")
    _write_marker(suite_dir / "running.marker")

    timings: dict[str, float] = {}
    commands: list[str] = []
    summaries: list[dict[str, Any]] = []
    log_path = suite_dir / "suite.log"

    def log(message: str) -> None:
        print(message)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(message + "\n")

    try:
        t0 = perf_counter()
        _dataset_preflight(root, configs[0].dataset_yaml)
        timings["dataset_verification"] = perf_counter() - t0
        t1 = perf_counter()
        calibration = validate_calibration_evidence(root, configs[0])
        timings["calibration_validation"] = perf_counter() - t1
        t2 = perf_counter()
        protocol_errors = validate_full_protocol_configs(configs)
        for cfg in configs:
            protocol_errors.extend(validate_full_config(cfg))
        if protocol_errors:
            raise ConfigError("; ".join(protocol_errors))
        timings["config_resolution"] = perf_counter() - t2
        if args.execute:
            _safe_gpu_preflight(configs[0])
        else:
            log("dry-run only: CUDA and execution checks are skipped")
        for cfg in configs:
            commands.append(_planned_command(cfg.model.family))
        _atomic_write(
            suite_dir / "protocol.json",
            {
                "dataset_yaml": str(configs[0].dataset_yaml),
                "dataset_yaml_sha256": sha256_file(configs[0].dataset_yaml),
                "calibration": calibration,
                "shared_fields": _EXPECTED_SHARED_KEYS,
                "test_split_prohibited": True,
                "selection_metric": "metrics/mAP50-95(B)",
                "train_split": "train",
                "validation_split": "val",
                "batch_size": configs[0].batch_size,
                "image_size": configs[0].image_size,
                "epochs": configs[0].epochs,
                "patience": configs[0].patience,
                "seed": configs[0].seed,
            },
        )
        _atomic_write(
            suite_dir / "environment.json",
            {
                "suite_id": suite_id,
                "timestamp_utc": now_utc(),
                "git_commit": git_commit(root),
                "git_dirty": git_dirty(root),
                "dependency_lock_sha256": dependency_lock_hash(root),
                "environment": to_jsonable(collect_environment_report(root)),
            },
        )
        _atomic_write(
            suite_dir / "provenance.json",
            {
                "suite_id": suite_id,
                "timestamp_utc": now_utc(),
                "models": models,
                "commands": commands,
                "calibration_reference": calibration.get("calibration_root"),
            },
        )
        _atomic_write(suite_dir / "commands.txt", "\n".join(commands) + "\n")
        _atomic_write(suite_dir / "calibration_reference.json", calibration)
        if args.dry_run and not args.execute:
            _write_suite_report(suite_dir, [], [])
            _write_marker(suite_dir / "completed.marker")
            return 0
        log("sequential full training on one RTX 4090")
        for cfg in configs:
            run_dir = _run_path_for(cfg)
            if args.skip_completed and (run_dir / "completed.marker").exists():
                if validate_completed_model_run(run_dir, cfg):
                    raise ConfigError(f"invalid completed run for skip-completed: {run_dir}")
                summaries.append(
                    summarize_model_run(cfg, run_dir, status="skipped", recovery_used=False)
                )
                continue
            if (
                run_dir.exists()
                and (run_dir / "completed.marker").exists()
                and not args.skip_completed
            ):
                raise ConfigError(f"completed run may not be overwritten: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_marker(run_dir / "planned.marker")
            _write_marker(run_dir / "running.marker")
            train_args = argparse.Namespace(execute=True, resume=args.resume)
            persist_run_metadata(run_dir, cfg, train_args)
            error: str | None = None
            recovery_used = False
            try:
                execute_training_with_args(cfg, run_dir, train_args)
            except Exception as exc:
                error = str(exc)
                if (run_dir / "raw").exists() and args.recover:
                    recover_training_artifacts(run_dir)
                    recovery_used = True
                    validation_errors = validate_completed_model_run(run_dir, cfg)
                    error = None
                    if validation_errors and not args.continue_on_error:
                        raise ConfigError("; ".join(validation_errors)) from None
                elif not args.continue_on_error:
                    raise
            if error is None and (run_dir / "completed.marker").exists():
                validation_errors = validate_completed_model_run(run_dir, cfg)
                if validation_errors and (run_dir / "raw").exists() and args.recover:
                    recover_training_artifacts(run_dir)
                    recovery_used = True
                    validation_errors = validate_completed_model_run(run_dir, cfg)
                    error = None
                if validation_errors:
                    error = "; ".join(validation_errors)
            if error:
                _write_marker(run_dir / "failed.marker")
                if not args.continue_on_error:
                    raise ConfigError(error)
                summaries.append(
                    summarize_model_run(
                        cfg, run_dir, status="failed", recovery_used=recovery_used, error=error
                    )
                )
                continue
            summaries.append(
                summarize_model_run(cfg, run_dir, status="completed", recovery_used=recovery_used)
            )
        diffs = validate_full_protocol_configs(configs)
        diffs.extend(_validate_effective_protocols(summaries))
        if diffs:
            raise ConfigError("; ".join(diffs))
        _write_suite_report(suite_dir, summaries, diffs)
        _atomic_write(
            suite_dir / "stage_timings.json",
            {
                "preflight": timings.get("dataset_verification", 0.0),
                "config_resolution": timings.get("config_resolution", 0.0),
                "calibration_validation": timings.get("calibration_validation", 0.0),
                "dataset_verification": timings.get("dataset_verification", 0.0),
                "report_generation": 0.0,
            },
        )
        _atomic_write(
            suite_dir / "model_runs.json",
            {"models": summaries},
        )
        if any(summary["status"] != "completed" for summary in summaries):
            _write_failure_marker(
                suite_dir / "failed.marker",
                reason="one or more full-suite models failed",
                summary={"models": summaries},
            )
            raise ConfigError("one or more full-suite models failed")
        _write_marker(suite_dir / "completed.marker")
        if args.print_report:
            print((suite_dir / "suite_report.md").read_text(encoding="utf-8"))
        return 0
    except Exception:
        _write_failure_marker(suite_dir / "failed.marker", reason="suite execution failed")
        if (suite_dir / "completed.marker").exists():
            (suite_dir / "completed.marker").unlink()
        raise
    finally:
        if (suite_dir / "running.marker").exists() and (suite_dir / "completed.marker").exists():
            (suite_dir / "running.marker").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--suite-id")
    parser.add_argument("--models")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        args.dry_run = True
    plan = {
        "suite_id": args.suite_id or default_suite_id(),
        "models": parse_models(args.models),
        "output_root": str(OUTPUT_ROOT),
        "execute": args.execute,
    }
    if args.dry_run and not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return run_suite(args)
