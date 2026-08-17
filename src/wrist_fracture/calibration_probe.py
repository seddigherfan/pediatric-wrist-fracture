from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil

from wrist_fracture.config import ExperimentConfig
from wrist_fracture.models.registry import resolve_model_spec


@dataclass(frozen=True)
class ProbeCacheKey:
    model_checkpoint_sha256: str | None
    image_size: int
    batch_size: int
    iterations: int
    amp: bool
    workers: int
    cache: str
    dataset_sha256: str | None
    training_config_sha256: str

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_probe_cache_key(cfg: ExperimentConfig, *, iterations: int) -> ProbeCacheKey:
    spec = resolve_model_spec(cfg.model)
    training_payload = {
        "model": cfg.model.family,
        "checkpoint": cfg.model.checkpoint,
        "scale": cfg.model.scale,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "augmentation": cfg.augmentation,
        "pretrained": cfg.pretrained,
        "seed": cfg.seed,
        "hardware": {
            "device": cfg.hardware.device,
            "amp": cfg.hardware.amp,
            "workers": cfg.hardware.workers,
            "cache": cfg.hardware.cache,
        },
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
    }
    return ProbeCacheKey(
        model_checkpoint_sha256=file_sha256(Path(spec.checkpoint)),
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        iterations=iterations,
        amp=cfg.hardware.amp,
        workers=cfg.hardware.workers,
        cache=cfg.hardware.cache,
        dataset_sha256=file_sha256(cfg.dataset_yaml),
        training_config_sha256=hashlib.sha256(
            json.dumps(training_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    )


def _cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _gpu_utilization() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"status": "unavailable", "reason": "cuda unavailable"}
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return {"status": "available", "value": float(util.gpu)}
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc)}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}


def _history_metrics(
    history_rows: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    latencies: list[float] = []
    for row in history_rows:
        for key in ("batch_time", "time", "elapsed", "duration"):
            if key in row and row[key] not in {None, ""}:
                try:
                    latencies.append(float(row[key]))
                    break
                except (TypeError, ValueError):
                    continue
    if not latencies:
        return None, None, None
    latencies.sort()
    mean_latency = sum(latencies) / len(latencies)
    median_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[min(len(latencies) - 1, int(round(len(latencies) * 0.95)) - 1)]
    return mean_latency, median_latency, p95_latency


def _update_stop(trainer: Any) -> None:
    if hasattr(trainer, "stop"):
        trainer.stop = True
    elif hasattr(trainer, "training"):
        trainer.training = False
    else:
        trainer.stop = True


def run_bounded_training_probe(
    cfg: ExperimentConfig,
    *,
    out_dir: Path,
    iterations: int = 30,
) -> dict[str, Any]:
    from ultralytics import YOLO

    from wrist_fracture.provenance import (
        collect_environment_report,
        git_commit,
        git_dirty,
        to_jsonable,
    )

    key = build_probe_cache_key(cfg, iterations=iterations)
    started = perf_counter()
    spec = resolve_model_spec(cfg.model)
    model = YOLO(spec.checkpoint)
    device = cfg.hardware.device
    warmup_iterations = max(1, min(5, iterations // 5))
    state = {
        "actual_iterations": 0,
        "warmup_elapsed": 0.0,
        "timed_elapsed": 0.0,
        "images_processed": 0,
        "batches_processed": 0,
    }
    peak_allocated = None
    peak_reserved = None
    failure: str | None = None
    status = "failed"
    last_start = started

    try:
        import torch

        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass

    def on_batch_start(_trainer: Any) -> None:
        nonlocal last_start
        last_start = perf_counter()

    def on_batch_end(trainer: Any) -> None:
        state["actual_iterations"] += 1
        state["batches_processed"] += 1
        elapsed = perf_counter() - last_start
        batch_images = int(getattr(trainer, "batch_size", cfg.batch_size) or cfg.batch_size)
        if state["actual_iterations"] <= warmup_iterations:
            state["warmup_elapsed"] += elapsed
        else:
            state["timed_elapsed"] += elapsed
            state["images_processed"] += batch_images
        if state["actual_iterations"] >= iterations:
            _update_stop(trainer)

    if hasattr(model, "add_callback"):
        model.add_callback("on_train_batch_start", on_batch_start)
        model.add_callback("on_train_batch_end", on_batch_end)

    train_kwargs: dict[str, Any] = {
        "data": str(cfg.dataset_yaml),
        "imgsz": 640,
        "epochs": 1,
        "patience": 1,
        "batch": cfg.batch_size,
        "workers": cfg.hardware.workers,
        "device": "0" if cfg.hardware.device.startswith("cuda") else cfg.hardware.device,
        "amp": cfg.hardware.amp,
        "seed": cfg.seed,
        "deterministic": cfg.hardware.deterministic,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "cache": cfg.hardware.cache,
        "save_period": 0,
        "project": str(out_dir / "raw"),
        "name": "probe",
        "exist_ok": True,
        "pretrained": cfg.pretrained,
        "plots": False,
        "val": False,
        "save_json": False,
        "resume": False,
        "verbose": False,
    }
    train_kwargs.update(cfg.augmentation)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    try:
        model.train(**train_kwargs)
        status = "stable" if state["actual_iterations"] >= iterations else "failed"
    except Exception as exc:
        failure = str(exc)
        status = "oom" if "out of memory" in failure.lower() else "failed"
    finally:
        try:
            import torch

            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
        except Exception:
            pass
        _cleanup_cuda()
        del model

    elapsed = perf_counter() - started
    actual_iterations = state["actual_iterations"]
    warmup_elapsed = state["warmup_elapsed"]
    timed_elapsed = max(state["timed_elapsed"], 1e-9)
    images_processed = state["images_processed"]
    batches_processed = state["batches_processed"]
    active_iterations = max(1, actual_iterations - warmup_iterations)
    images_per_sec = images_processed / timed_elapsed if images_processed else None
    batches_per_sec = batches_processed / timed_elapsed if batches_processed else None
    mean_latency, median_latency, p95_latency = _history_metrics([])
    telemetry = psutil.virtual_memory()
    result = {
        "cache_key": key.digest(),
        "cache_key_components": key.__dict__,
        "model_family": cfg.model.family,
        "batch_size": cfg.batch_size,
        "status": status,
        "cuda_oom": status == "oom",
        "exception": failure,
        "actual_optimizer_iterations": actual_iterations,
        "warmup_iterations": warmup_iterations,
        "warmup_elapsed_s": warmup_elapsed,
        "iteration_mean_latency_s": (
            mean_latency if mean_latency is not None else timed_elapsed / active_iterations
        ),
        "iteration_median_latency_s": median_latency,
        "iteration_p95_latency_s": p95_latency,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "images_per_sec": images_per_sec,
        "batches_per_sec": batches_per_sec,
        "elapsed_s": elapsed,
        "estimated_epoch_duration_s": None if not batches_per_sec else cfg.epochs / batches_per_sec,
        "cpu_utilization_pct": psutil.cpu_percent(interval=None),
        "ram_usage_bytes": telemetry.used,
        "gpu_utilization": _gpu_utilization(),
        "config": to_jsonable(cfg),
        "environment": to_jsonable(collect_environment_report(Path.cwd())),
        "git_commit": git_commit(Path.cwd()),
        "git_dirty": git_dirty(Path.cwd()),
    }
    write_atomic(out_dir / "probe_result.json", json.dumps(result, indent=2, sort_keys=True))
    return result
