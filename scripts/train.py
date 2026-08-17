from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from wrist_fracture.config import (
    ConfigError,
    ExperimentConfig,
    config_to_dict,
    load_config_bundle,
    validate_experiment_config,
)
from wrist_fracture.models.registry import describe_model_spec, resolve_model_spec
from wrist_fracture.provenance import (
    collect_environment_report,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    sha256_file,
    to_jsonable,
)
from wrist_fracture.runtime import ExecutionOptions


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_run_id(cfg: ExperimentConfig) -> str:
    return (
        cfg.run.run_id
        or f"{cfg.run.name}-{_now().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
    )


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    return load_config_bundle(
        args.config,
        model_path=args.model_config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )


def run_root(cfg: ExperimentConfig, run_id: str) -> Path:
    return cfg.run.output_root / cfg.model.family / run_id


def ensure_unique_run_dir(path: Path, *, resume: bool = False) -> None:
    if path.exists():
        if resume and (path / "completed.marker").exists():
            raise ConfigError(f"cannot resume completed run: {path}")
        if not resume:
            raise ConfigError(f"run directory already exists: {path}")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def persist_run_metadata(root: Path, cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    env = collect_environment_report(Path.cwd())
    spec = resolve_model_spec(cfg.model)
    payload = {
        "timestamp_utc": _now(),
        "git_commit": git_commit(Path.cwd()),
        "git_dirty": git_dirty(Path.cwd()),
        "dependency_lock_sha256": dependency_lock_hash(Path.cwd()),
        "dataset_yaml_sha256": sha256_file(cfg.dataset_yaml) if cfg.dataset_yaml.exists() else None,
        "dataset_split_yaml_sha256": sha256_file(cfg.dataset_split_yaml)
        if cfg.dataset_split_yaml and cfg.dataset_split_yaml.exists()
        else None,
        "command": sys.argv,
        "model": describe_model_spec(spec, imgsz=cfg.image_size),
        "environment": to_jsonable(env),
        "config": config_to_dict(cfg),
    }
    write_atomic(
        root / "resolved_config.yaml",
        json.dumps(config_to_dict(cfg), indent=2, sort_keys=True),
    )
    write_atomic(root / "environment.json", json.dumps(to_jsonable(env), indent=2, sort_keys=True))
    write_atomic(root / "provenance.json", json.dumps(payload, indent=2, sort_keys=True))
    write_atomic(root / "command.txt", " ".join(sys.argv))
    for folder in ["logs", "checkpoints", "raw", "metrics", "figures", "benchmarks"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "started.marker").write_text(_now(), encoding="utf-8")


def finalize_run(root: Path, success: bool) -> None:
    marker = root / ("completed.marker" if success else "interrupted.marker")
    marker.write_text(_now(), encoding="utf-8")


def dry_plan(cfg: ExperimentConfig, run_id: str) -> dict[str, object]:
    spec = resolve_model_spec(cfg.model)
    return {
        "run_id": run_id,
        "run_root": str(run_root(cfg, run_id)),
        "model": describe_model_spec(spec, imgsz=cfg.image_size),
        "config": config_to_dict(cfg),
        "python": sys.version,
    }


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
        if device == "cuda":
            return torch.cuda.device_count() > 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        else:
            index = 0
        return index < torch.cuda.device_count()
    except Exception:
        return False


def _normalize_device(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return device


def _maybe_copy_or_link(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)
    return dst


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _is_nan_or_inf(value: Any) -> bool:
    try:
        import math

        if isinstance(value, float):
            return math.isnan(value) or math.isinf(value)
    except Exception:
        pass
    return False


def _json_safe_number(value: Any) -> float | int | None:
    parsed = _to_float(value)
    if parsed is None or _is_nan_or_inf(parsed):
        return None
    if float(parsed).is_integer():
        return int(parsed)
    return parsed


def _validate_results_csv_columns(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ConfigError("results.csv is empty")
    expected = {
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    }
    missing = sorted(expected.difference(rows[0].keys()))
    if missing:
        raise ConfigError(f"results.csv missing expected columns: {', '.join(missing)}")


def _serialize_public_metric(value: Any) -> dict[str, Any] | None:
    public: dict[str, Any] = {}
    for attr in ("mp", "mr", "map50", "map", "fitness"):
        if hasattr(value, attr):
            public[attr] = _json_safe_number(getattr(value, attr))
    for attr in ("p", "r", "f1", "maps", "ap", "ap50", "ap75", "nc", "nt_per_class"):
        if hasattr(value, attr):
            public[attr] = _normalize_json_value(getattr(value, attr))
    if hasattr(value, "speed"):
        public["speed"] = _normalize_json_value(value.speed)
    if hasattr(value, "results_dict"):
        public["results_dict"] = _normalize_json_value(value.results_dict)
    if hasattr(value, "summary"):
        summary = value.summary
        if callable(summary):
            try:
                public["summary"] = _normalize_json_value(summary())
            except TypeError:
                pass
    if public:
        public["type"] = type(value).__name__
        return public
    return None


def _normalize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if _is_nan_or_inf(value) else value
    if isinstance(value, Path):
        return str(value)
    metric_payload = _serialize_public_metric(value)
    if metric_payload is not None:
        return metric_payload
    if dataclasses.is_dataclass(value):
        normalized = _normalize_json_value(dataclasses.asdict(value))
        if isinstance(normalized, dict):
            normalized["type"] = type(value).__name__
        return normalized
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _normalize_json_value(value.item())
        if isinstance(value, np.ndarray):
            return [_normalize_json_value(item) for item in value.tolist()]
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return _normalize_json_value(value.item())
            return [_normalize_json_value(item) for item in value.detach().cpu().tolist()]
    except Exception:
        pass
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        public = {
            key: _normalize_json_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public:
            public["type"] = type(value).__name__
            return public
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return str(value)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_checkpoint_paths(save_dir: Path) -> dict[str, Path | None]:
    weights = save_dir / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    return {
        "best": best if best.exists() else None,
        "last": last if last.exists() else None,
    }


def _normalize_history(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            parsed = _to_float(value)
            item[key] = parsed if parsed is not None else value
        normalized.append(item)
    return normalized


def _schema_versioned_validation_payload(
    *,
    metrics: dict[str, Any],
    save_dir: Path,
    raw_results: Any,
) -> dict[str, Any]:
    precision = _json_safe_number(metrics.get("final_precision"))
    recall = _json_safe_number(metrics.get("final_recall"))
    map50 = _json_safe_number(metrics.get("final_map50"))
    map50_95 = _json_safe_number(metrics.get("final_map50_95"))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    payload = {
        "schema_version": 1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "fitness": _json_safe_number(getattr(raw_results, "fitness", None))
        if raw_results is not None
        else None,
        "per_class_ap": _normalize_json_value(getattr(raw_results, "maps", None))
        if raw_results is not None
        else None,
        "speed": _normalize_json_value(getattr(raw_results, "speed", None))
        if raw_results is not None
        else None,
        "raw_results": _normalize_json_value(raw_results),
        "ultralytics_save_dir": str(save_dir),
    }
    return payload


def _metrics_from_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    final = rows[-1]
    best = max(
        rows,
        key=lambda row: _to_float(row.get("metrics/mAP50-95(B)")) or float("-inf"),
    )
    return {
        "final_precision": _to_float(final.get("metrics/precision(B)")),
        "final_recall": _to_float(final.get("metrics/recall(B)")),
        "final_map50": _to_float(final.get("metrics/mAP50(B)")),
        "final_map50_95": _to_float(final.get("metrics/mAP50-95(B)")),
        "best_epoch": int(_to_float(best.get("epoch")) or 0),
        "best_map50_95": _to_float(best.get("metrics/mAP50-95(B)")),
    }


def _recover_training_artifacts(root: Path) -> None:
    raw_candidates = sorted((root / "raw").glob("**/results.csv"))
    if not raw_candidates:
        raise ConfigError("no raw Ultralytics results.csv found for recovery")
    history_src = raw_candidates[0]
    save_dir = history_src.parent
    rows = _read_csv_rows(history_src)
    _validate_results_csv_columns(rows)
    history_rows = _normalize_history(rows)
    _write_csv_rows(root / "metrics" / "history.csv", history_rows)
    metrics = _metrics_from_history(history_rows)
    checkpoints = _collect_checkpoint_paths(save_dir)
    best_src = checkpoints["best"]
    last_src = checkpoints["last"]
    if best_src is None or last_src is None:
        raise ConfigError("missing expected Ultralytics checkpoints")
    actual_best = _maybe_copy_or_link(best_src, root / "checkpoints" / "best.pt")
    actual_last = _maybe_copy_or_link(last_src, root / "checkpoints" / "last.pt")
    validation_payload = _schema_versioned_validation_payload(
        metrics=metrics,
        save_dir=save_dir,
        raw_results=SimpleNamespace(
            fitness=metrics.get("best_map50_95"),
            maps=metrics.get("best_map50_95"),
            speed=None,
        ),
    )
    _write_validation_json(root, validation_payload)
    summary_payload = {
        "run_status": "completed",
        "started_at": None,
        "ended_at": _now(),
        "duration_seconds": None,
        "gpu_peak_memory_bytes": None,
        "checkpoint_paths": {"best": str(actual_best), "last": str(actual_last)},
        "ultralytics_save_dir": str(save_dir),
        "config": None,
        "execute": True,
        "smoke": True,
        **metrics,
    }
    write_atomic(
        root / "metrics" / "run_summary.json",
        json.dumps(_normalize_json_value(summary_payload), indent=2, sort_keys=True),
    )
    interrupted = root / "interrupted.marker"
    if interrupted.exists():
        interrupted.unlink()
    (root / "completed.marker").write_text(_now(), encoding="utf-8")


def _write_validation_json(root: Path, payload: dict[str, Any]) -> None:
    write_atomic(
        root / "metrics" / "validation.json",
        json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True),
    )


def _write_run_summary(
    root: Path,
    *,
    cfg: ExperimentConfig,
    options: ExecutionOptions,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    save_dir: Path,
    history_rows: list[dict[str, Any]],
    checkpoints: dict[str, Path | None],
    gpu_peak_memory_bytes: int | None,
    effective_protocol: dict[str, Any] | None = None,
) -> None:
    metrics = _metrics_from_history(history_rows)
    payload = {
        "run_status": "completed",
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
        "checkpoint_paths": {
            "best": str(checkpoints["best"]) if checkpoints["best"] else None,
            "last": str(checkpoints["last"]) if checkpoints["last"] else None,
        },
        "ultralytics_save_dir": str(save_dir),
        "config": config_to_dict(cfg),
        "execute": options.execute,
        "smoke": options.smoke,
        "effective_protocol": _normalize_json_value(effective_protocol),
        **metrics,
    }
    write_atomic(
        root / "metrics" / "run_summary.json",
        json.dumps(_normalize_json_value(payload), indent=2, sort_keys=True),
    )


def _recover_from_run_root(root: Path) -> None:
    _recover_training_artifacts(root)


def _execute_training(cfg: ExperimentConfig, root: Path) -> None:
    raise NotImplementedError("use _execute_training_with_args")


def _execute_training_with_args(
    cfg: ExperimentConfig, root: Path, args: argparse.Namespace
) -> None:
    from ultralytics import YOLO

    started_at = _now()
    start_perf = perf_counter()
    gpu_peak_memory_bytes: int | None = None
    device = _normalize_device(cfg.hardware.device)
    spec = resolve_model_spec(cfg.model)
    model = YOLO(spec.checkpoint)
    train_kwargs: dict[str, Any] = {
        "data": str(cfg.dataset_yaml),
        "imgsz": cfg.image_size,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "batch": cfg.batch_size,
        "workers": cfg.hardware.workers,
        "device": device,
        "amp": cfg.hardware.amp,
        "seed": cfg.seed,
        "deterministic": cfg.hardware.deterministic,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "cache": cfg.hardware.cache,
        "save_period": cfg.run.save_period,
        "project": str((root / "raw").resolve()),
        "name": "train",
        "exist_ok": True,
        "pretrained": cfg.pretrained,
        "plots": True,
        "val": True,
        "save_json": cfg.save_json,
        "resume": bool(cfg.run.resume),
    }
    train_kwargs.update(cfg.augmentation)
    if cfg.resume_checkpoint is not None:
        train_kwargs["resume"] = str(cfg.resume_checkpoint)
    if cfg.run.validation_split != "val":
        raise ConfigError("validation split must remain val during training")
    if cfg.run.test_split == "val":
        raise ConfigError("test split must not be used during training")
    if cfg.hardware.device == "cpu" and not cfg.hardware.allow_cpu_training:
        raise ConfigError("CPU full training is disabled")
    if cfg.run.resume and cfg.resume_checkpoint is None:
        raise ConfigError("safe resume validation failed")
    if cfg.run.resume and not cfg.resume_checkpoint.exists():
        raise ConfigError("safe resume validation failed")
    if device == "cpu" and not cfg.hardware.allow_cpu_training:
        raise ConfigError("no CPU full training")
    if device == "cpu" and cfg.epochs > 1:
        raise ConfigError("no CPU full training")
    try:
        results = model.train(**train_kwargs)
        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", root / "raw" / "train"))
        history_src = save_dir / "results.csv"
        history_rows_raw = _read_csv_rows(history_src) if history_src.exists() else []
        _validate_results_csv_columns(history_rows_raw)
        history_rows = _normalize_history(history_rows_raw)
        _write_csv_rows(root / "metrics" / "history.csv", history_rows)
        metrics = _metrics_from_history(history_rows)
        checkpoints = _collect_checkpoint_paths(save_dir)
        best_src = checkpoints["best"]
        last_src = checkpoints["last"]
        best_dst = root / "checkpoints" / "best.pt"
        last_dst = root / "checkpoints" / "last.pt"
        actual_best = _maybe_copy_or_link(best_src, best_dst) if best_src else None
        actual_last = _maybe_copy_or_link(last_src, last_dst) if last_src else None
        if actual_best is None or actual_last is None:
            raise ConfigError("missing expected Ultralytics checkpoints")
        trainer_args = getattr(getattr(results, "trainer", None), "args", None)
        effective_protocol = {
            "optimizer": getattr(trainer_args, "optimizer", None) or train_kwargs.get("optimizer"),
            "lr0": getattr(trainer_args, "lr0", None) or train_kwargs.get("lr0"),
            "momentum": getattr(trainer_args, "momentum", None),
            "augmentation": {
                "RandAugment": getattr(trainer_args, "auto_augment", None),
                "erasing": getattr(trainer_args, "erasing", None) or train_kwargs.get("erasing"),
                "horizontal_flip": getattr(trainer_args, "fliplr", None)
                or train_kwargs.get("fliplr"),
                "translate": getattr(trainer_args, "translate", None)
                or train_kwargs.get("translate"),
                "scale": getattr(trainer_args, "scale", None) or train_kwargs.get("scale"),
            },
        }
        gpu_peak_memory_bytes = None
        try:
            import torch

            if torch.cuda.is_available() and device != "cpu":
                gpu_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            gpu_peak_memory_bytes = None
        validation_payload = {
            "schema_version": 1,
            "precision": metrics.get("final_precision"),
            "recall": metrics.get("final_recall"),
            "f1": None,
            "map50": metrics.get("final_map50"),
            "map50_95": metrics.get("final_map50_95"),
            "best_epoch": metrics.get("best_epoch"),
            "best_map50_95": metrics.get("best_map50_95"),
            "fitness": _json_safe_number(getattr(results, "fitness", None)),
            "per_class_ap": _normalize_json_value(getattr(results, "maps", None)),
            "speed": _normalize_json_value(getattr(results, "speed", None)),
            "raw_results": _normalize_json_value(results),
        }
        if validation_payload["precision"] is not None and validation_payload["recall"] is not None:
            p = validation_payload["precision"]
            r = validation_payload["recall"]
            if p + r != 0:
                validation_payload["f1"] = 2 * p * r / (p + r)
        _write_validation_json(root, validation_payload)
        ended_at = _now()
        duration_seconds = perf_counter() - start_perf
        _write_run_summary(
            root,
            cfg=cfg,
            options=ExecutionOptions(execute=args.execute, smoke=args.smoke),
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            save_dir=save_dir,
            history_rows=history_rows,
            checkpoints={"best": actual_best, "last": actual_last},
            gpu_peak_memory_bytes=gpu_peak_memory_bytes,
            effective_protocol=effective_protocol,
        )
        (root / "completed.marker").write_text(_now(), encoding="utf-8")
    except Exception:
        (root / "interrupted.marker").write_text(_now(), encoding="utf-8")
        raise


SMOKE_SAFETY_CAPS = {
    "image_size": 320,
    "epochs": 1,
    "batch_size": 4,
    "patience": 1,
    "run.repeated_runs": 1,
}


def _validate_smoke_caps(cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    if cfg.image_size > SMOKE_SAFETY_CAPS["image_size"]:
        errors.append("smoke image_size exceeds safety cap")
    if cfg.epochs > SMOKE_SAFETY_CAPS["epochs"]:
        errors.append("smoke epochs exceeds safety cap")
    if cfg.batch_size > SMOKE_SAFETY_CAPS["batch_size"]:
        errors.append("smoke batch_size exceeds safety cap")
    if cfg.patience > SMOKE_SAFETY_CAPS["patience"]:
        errors.append("smoke patience exceeds safety cap")
    if cfg.run.repeated_runs > SMOKE_SAFETY_CAPS["run.repeated_runs"]:
        errors.append("smoke repeated_runs exceeds safety cap")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-postprocessing", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    args = parser.parse_args()

    if getattr(args, "recover_postprocessing", False):
        if not args.run_root:
            raise ConfigError("--run-root is required with --recover-postprocessing")
        _recover_from_run_root(Path(args.run_root))
        return

    cfg = resolve_config(args)
    if args.smoke:
        cfg = ExperimentConfig(
            dataset_yaml=cfg.dataset_yaml,
            dataset_split_yaml=cfg.dataset_split_yaml,
            model=cfg.model,
            hardware=cfg.hardware,
            run=cfg.run,
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
    if not args.execute and not (args.preflight or args.dry_run):
        raise ConfigError("full training requires explicit --execute")
    gpu_ready = _cuda_is_available() if args.execute else None
    if args.print_resolved_config:
        print(json.dumps(config_to_dict(cfg), indent=2, sort_keys=True))
    errors = validate_experiment_config(
        cfg,
        dry_run=not args.execute,
        allow_cpu_smoke=args.allow_cpu_smoke and args.smoke,
    )
    if args.smoke:
        errors.extend(_validate_smoke_caps(cfg))
    if errors:
        raise ConfigError("; ".join(errors))
    run_id = build_run_id(cfg)
    root = run_root(cfg, run_id)
    if args.resume and not root.exists():
        raise ConfigError("resume requested but run directory does not exist")
    ensure_unique_run_dir(root, resume=args.resume)
    if cfg.hardware.device.startswith("cuda") and args.execute:
        if not gpu_ready:
            raise ConfigError("torch.cuda.is_available() is false")
        if not _cuda_device_exists(cfg.hardware.device):
            raise ConfigError("requested CUDA device does not exist")
    if args.preflight or args.dry_run or not args.execute:
        print(json.dumps(dry_plan(cfg, run_id), indent=2, sort_keys=True))
        return
    persist_run_metadata(root, cfg, args)
    try:
        _execute_training_with_args(cfg, root, args)
    except Exception:
        finalize_run(root, success=False)
        raise


if __name__ == "__main__":
    main()
