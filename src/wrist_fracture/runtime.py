from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from wrist_fracture.config import ConfigError, ExperimentConfig, config_to_dict
from wrist_fracture.models.registry import describe_model_spec, resolve_model_spec
from wrist_fracture.provenance import (
    collect_environment_report,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    sha256_file,
    to_jsonable,
)


@dataclasses.dataclass(frozen=True)
class ExecutionOptions:
    execute: bool = False
    smoke: bool = False


def now_utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_run_id(cfg: ExperimentConfig) -> str:
    import uuid

    return (
        cfg.run.run_id
        or f"{cfg.run.name}-{now_utc().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
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


def normalize_device(device: str | int) -> str:
    if isinstance(device, int):
        return str(device)
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return device


def normalize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): normalize_json_value(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_json_value(item) for item in value]
    if dataclasses.is_dataclass(value):
        normalized = normalize_json_value(dataclasses.asdict(value))
        if isinstance(normalized, dict):
            normalized["type"] = type(value).__name__
        return normalized
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return normalize_json_value(value.item())
        if isinstance(value, np.ndarray):
            return [normalize_json_value(item) for item in value.tolist()]
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return normalize_json_value(value.item())
            return [normalize_json_value(item) for item in value.detach().cpu().tolist()]
    except Exception:
        pass
    if hasattr(value, "__dict__"):
        public = {
            key: normalize_json_value(item)
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


def _maybe_copy_or_link(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)
    return dst


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
    return {"best": best if best.exists() else None, "last": last if last.exists() else None}


def _find_ultralytics_save_dir(raw_root: Path) -> Path:
    candidates = sorted(raw_root.glob("**/results.csv"))
    if not candidates:
        raise ConfigError("no raw Ultralytics results.csv found for recovery")
    return candidates[0].parent


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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _json_safe_number(value: Any) -> float | int | None:
    parsed = _to_float(value)
    if parsed is None or parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    if float(parsed).is_integer():
        return int(parsed)
    return parsed


def _normalize_history(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            parsed = _to_float(value)
            item[key] = parsed if parsed is not None else value
        normalized.append(item)
    return normalized


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
    return {
        "schema_version": 1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "fitness": _json_safe_number(getattr(raw_results, "fitness", None))
        if raw_results is not None
        else None,
        "per_class_ap": normalize_json_value(getattr(raw_results, "maps", None))
        if raw_results is not None
        else None,
        "speed": normalize_json_value(getattr(raw_results, "speed", None))
        if raw_results is not None
        else None,
        "raw_results": normalize_json_value(raw_results),
        "ultralytics_save_dir": str(save_dir),
    }


def write_validation_json(root: Path, payload: dict[str, Any]) -> None:
    write_atomic(
        root / "metrics" / "validation.json",
        json.dumps(normalize_json_value(payload), indent=2, sort_keys=True),
    )


def write_run_summary(
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
        "effective_protocol": normalize_json_value(effective_protocol),
        **metrics,
    }
    write_atomic(
        root / "metrics" / "run_summary.json",
        json.dumps(normalize_json_value(payload), indent=2, sort_keys=True),
    )


def recover_training_artifacts(root: Path) -> None:
    save_dir = _find_ultralytics_save_dir(root / "raw")
    history_src = save_dir / "results.csv"
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
    write_validation_json(root, validation_payload)
    summary_payload = {
        "run_status": "completed",
        "started_at": None,
        "ended_at": now_utc(),
        "duration_seconds": None,
        "gpu_peak_memory_bytes": None,
        "checkpoint_paths": {"best": str(actual_best), "last": str(actual_last)},
        "ultralytics_save_dir": str(save_dir),
        "config": None,
        "execute": True,
        "smoke": False,
        "effective_protocol": None,
        **metrics,
    }
    write_atomic(
        root / "metrics" / "run_summary.json",
        json.dumps(normalize_json_value(summary_payload), indent=2, sort_keys=True),
    )
    interrupted = root / "interrupted.marker"
    if interrupted.exists():
        interrupted.unlink()
    (root / "completed.marker").write_text(now_utc(), encoding="utf-8")


def _capture_effective_protocol(results: Any, train_kwargs: dict[str, Any]) -> dict[str, Any]:
    trainer = getattr(results, "trainer", None)
    args_obj = getattr(trainer, "args", None)
    effective = {
        "optimizer": getattr(args_obj, "optimizer", None),
        "lr0": getattr(args_obj, "lr0", None),
        "momentum": getattr(args_obj, "momentum", None),
        "augmentation": {
            "RandAugment": getattr(args_obj, "auto_augment", None),
            "erasing": getattr(args_obj, "erasing", None),
            "horizontal_flip": getattr(args_obj, "fliplr", None),
            "translate": getattr(args_obj, "translate", None),
            "scale": getattr(args_obj, "scale", None),
        },
    }
    if not effective["optimizer"]:
        effective["optimizer"] = train_kwargs.get("optimizer")
    if effective["lr0"] is None:
        effective["lr0"] = train_kwargs.get("lr0")
    if effective["augmentation"]["horizontal_flip"] is None:
        effective["augmentation"]["horizontal_flip"] = train_kwargs.get("fliplr")
    if effective["augmentation"]["translate"] is None:
        effective["augmentation"]["translate"] = train_kwargs.get("translate")
    if effective["augmentation"]["scale"] is None:
        effective["augmentation"]["scale"] = train_kwargs.get("scale")
    if effective["augmentation"]["erasing"] is None:
        effective["augmentation"]["erasing"] = train_kwargs.get("erasing")
    if effective["augmentation"]["RandAugment"] is None:
        effective["augmentation"]["RandAugment"] = train_kwargs.get("auto_augment")
    return effective


def persist_run_metadata(root: Path, cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    env = collect_environment_report(Path.cwd())
    spec = resolve_model_spec(cfg.model)
    payload = {
        "timestamp_utc": now_utc(),
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
        root / "resolved_config.yaml", json.dumps(config_to_dict(cfg), indent=2, sort_keys=True)
    )
    write_atomic(root / "environment.json", json.dumps(to_jsonable(env), indent=2, sort_keys=True))
    write_atomic(root / "provenance.json", json.dumps(payload, indent=2, sort_keys=True))
    write_atomic(root / "command.txt", " ".join(sys.argv))
    for folder in ["logs", "checkpoints", "raw", "metrics", "figures", "benchmarks"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "started.marker").write_text(now_utc(), encoding="utf-8")


def execute_training_with_args(cfg: ExperimentConfig, root: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    started_at = now_utc()
    device = normalize_device(cfg.hardware.device)
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
        effective_protocol = _capture_effective_protocol(results, train_kwargs)
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
            "per_class_ap": normalize_json_value(getattr(results, "maps", None)),
            "speed": normalize_json_value(getattr(results, "speed", None)),
            "raw_results": normalize_json_value(results),
        }
        if validation_payload["precision"] is not None and validation_payload["recall"] is not None:
            p = validation_payload["precision"]
            r = validation_payload["recall"]
            if p + r != 0:
                validation_payload["f1"] = 2 * p * r / (p + r)
        write_validation_json(root, validation_payload)
        ended_at = now_utc()
        duration_seconds = 0.0
        write_run_summary(
            root,
            cfg=cfg,
            options=ExecutionOptions(
                execute=bool(getattr(args, "execute", False)),
                smoke=bool(getattr(args, "smoke", False)),
            ),
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            save_dir=save_dir,
            history_rows=history_rows,
            checkpoints={"best": actual_best, "last": actual_last},
            gpu_peak_memory_bytes=gpu_peak_memory_bytes,
            effective_protocol=effective_protocol,
        )
        (root / "completed.marker").write_text(now_utc(), encoding="utf-8")
    except Exception:
        (root / "interrupted.marker").write_text(now_utc(), encoding="utf-8")
        raise
