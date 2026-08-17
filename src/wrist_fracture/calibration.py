from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import yaml

from wrist_fracture.calibration_probe import run_bounded_training_probe
from wrist_fracture.config import ExperimentConfig, config_to_dict
from wrist_fracture.models.registry import describe_model_spec, resolve_model_spec
from wrist_fracture.provenance import (
    collect_environment_report,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    to_jsonable,
)

CALIBRATION_BATCH_CANDIDATES = (8, 16, 32, 48, 64)
CALIBRATION_MAX_BATCHES = 3
CALIBRATED_MODEL_ORDER = ("yolov8", "yolov9", "yolo26")
CALIBRATION_SCHEMA_VERSION = 2


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def candidate_batches(
    max_batch: int | None = None, *, candidates: list[int] | tuple[int, ...] | None = None
) -> list[int]:
    batches = list(candidates or CALIBRATION_BATCH_CANDIDATES)
    if max_batch is not None:
        batches = [batch for batch in batches if batch <= max_batch]
    batches = sorted(dict.fromkeys(int(batch) for batch in batches))
    return batches


def select_common_batch(results: dict[str, dict[int, bool]]) -> int | None:
    common = [
        batch
        for batch in CALIBRATION_BATCH_CANDIDATES
        if all(model_results.get(batch) for model_results in results.values())
    ]
    return max(common) if common else None


def select_recommended_batch(candidates: list[dict[str, Any]]) -> int | None:
    stable = [int(row["batch_size"]) for row in candidates if row.get("status") == "stable"]
    return max(stable) if stable else None


def select_largest_stable_batch(candidates: list[dict[str, Any]]) -> int | None:
    stable = [int(row["batch_size"]) for row in candidates if row.get("status") == "stable"]
    return max(stable) if stable else None


def select_largest_common_stable_batch(
    results: dict[str, list[dict[str, Any]]],
) -> int | None:
    per_model = [select_largest_stable_batch(rows) for rows in results.values()]
    stable = [batch for batch in per_model if batch is not None]
    return min(stable) if stable and len(stable) == len(per_model) else None


def recommended_hardware_config(cfg: ExperimentConfig, *, batch_size: int) -> dict[str, Any]:
    return {
        "device": cfg.hardware.device,
        "amp": cfg.hardware.amp,
        "workers": cfg.hardware.workers,
        "cache": cfg.hardware.cache,
        "deterministic": True,
        "persistent_workers": bool(cfg.hardware.workers > 0),
        "pin_memory": cfg.hardware.device.startswith("cuda"),
        "prefetch_factor": 2 if cfg.hardware.workers > 0 else None,
        "batch_size": batch_size,
    }


def freeze_full_config(cfg: ExperimentConfig, *, batch_size: int) -> dict[str, Any]:
    payload = config_to_dict(cfg)
    payload["batch_size"] = batch_size
    payload["hardware"] = {
        **payload["hardware"],
        "deterministic": True,
        "persistent_workers": bool(cfg.hardware.workers > 0),
        "pin_memory": cfg.hardware.device.startswith("cuda"),
        "prefetch_factor": 2 if cfg.hardware.workers > 0 else None,
    }
    return payload


def build_command(*, model_path: Path, hardware_path: Path, run_path: Path, execute: bool) -> str:
    return " ".join(
        [
            "uv run python scripts/train.py",
            "--config configs/experiment.yaml",
            f"--model-config {model_path.as_posix()}",
            f"--hardware-config {hardware_path.as_posix()}",
            f"--run-config {run_path.as_posix()}",
            "--execute" if execute else "--dry-run",
            "--smoke",
        ]
    )


def build_execution_command(
    *, model_path: Path, hardware_path: Path, run_path: Path, batch_size: int
) -> str:
    return " ".join(
        [
            "uv run python scripts/train.py",
            "--config configs/experiment.yaml",
            f"--model-config {model_path.as_posix()}",
            f"--hardware-config {hardware_path.as_posix()}",
            f"--run-config {run_path.as_posix()}",
            "--execute",
            "--smoke",
            f"--batch-size {batch_size}",
        ]
    )


def build_hardware_profile(cfg: ExperimentConfig) -> dict[str, Any]:
    env = collect_environment_report(Path.cwd())
    spec = resolve_model_spec(cfg.model)
    return {
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(Path.cwd()),
        "git_dirty": git_dirty(Path.cwd()),
        "dependency_lock_sha256": dependency_lock_hash(Path.cwd()),
        "environment": to_jsonable(env),
        "model": describe_model_spec(spec, imgsz=cfg.image_size),
    }


def build_calibration_report(
    *,
    cfg: ExperimentConfig,
    model_family: str,
    candidates: list[dict[str, Any]],
    recommended_batch_size: int | None,
    applied: bool,
    resume_state: dict[str, Any] | None,
) -> dict[str, Any]:
    spec = resolve_model_spec(cfg.model)
    return {
        "schema_version": 1,
        "timestamp_utc": now_utc(),
        "model_family": model_family,
        "resolved_model": describe_model_spec(spec, imgsz=cfg.image_size),
        "stable_candidate_count": sum(1 for row in candidates if row.get("status") == "stable"),
        "recommended_batch_size": recommended_batch_size,
        "common_stable_batch": recommended_batch_size,
        "recommended_protocol": recommended_hardware_config(
            cfg, batch_size=recommended_batch_size or cfg.batch_size
        ),
        "resume_state": resume_state,
        "applied": applied,
        "candidates": candidates,
        "config": config_to_dict(cfg),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return to_jsonable(value)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def execution_root(out_dir: Path, execution_id: str) -> Path:
    return out_dir / "executions" / execution_id


def execution_id_for_report(report: dict[str, Any]) -> str:
    digest_source = json.dumps(_json_safe(report), sort_keys=True).encode("utf-8")
    return hashlib.sha256(digest_source).hexdigest()[:16]


def report_provenance_kind(report: dict[str, Any]) -> str:
    candidate_statuses = {row.get("status") for row in report.get("candidates", [])}
    if candidate_statuses <= {"planned"}:
        return "planned"
    if candidate_statuses & {"stable", "oom", "failed", "skipped", "skipped_after_oom"}:
        return "real"
    return "unknown"


def report_has_complete_real_evidence(report: dict[str, Any]) -> bool:
    candidates = report.get("candidates", [])
    if not candidates:
        return False
    if report.get("common_stable_batch") is None:
        return False
    if report.get("stable_candidate_count", 0) <= 0:
        return False
    if any(row.get("status") == "planned" for row in candidates):
        return False
    model_families = {row.get("model_family") for row in candidates}
    if report.get("model_family") == "all":
        if not {"yolov8", "yolov9", "yolo26"}.issubset(model_families):
            return False
        for model_family in CALIBRATED_MODEL_ORDER:
            rows = [row for row in candidates if row.get("model_family") == model_family]
            if not any(row.get("status") == "stable" for row in rows):
                return False
    else:
        model_family = report.get("model_family")
        rows = [row for row in candidates if row.get("model_family") == model_family]
        if not rows:
            return False
        if not any(row.get("status") == "stable" for row in rows):
            return False
    return True


def write_reports(
    out_dir: Path,
    *,
    report: dict[str, Any],
    hardware_profile: dict[str, Any],
    stage_timings: dict[str, Any],
    environment: dict[str, Any],
    commands: list[str],
    recommended_config: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        out_dir / "calibration_report.json",
        json.dumps(_json_safe(report), indent=2, sort_keys=True),
    )
    rows = report.get("candidates", [])
    with (out_dir / "calibration_report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    write_atomic(
        out_dir / "hardware_profile.json",
        json.dumps(_json_safe(hardware_profile), indent=2, sort_keys=True),
    )
    write_atomic(
        out_dir / "stage_timings.json",
        json.dumps(_json_safe(stage_timings), indent=2, sort_keys=True),
    )
    write_atomic(
        out_dir / "environment.json",
        json.dumps(_json_safe(environment), indent=2, sort_keys=True),
    )
    write_atomic(out_dir / "commands.txt", "\n".join(commands) + ("\n" if commands else ""))
    write_atomic(
        out_dir / "recommended_full_config.yaml",
        yaml.safe_dump({"experiment": recommended_config}, sort_keys=False),
    )


def write_execution_artifacts(
    out_dir: Path,
    *,
    execution_id: str,
    report: dict[str, Any],
    hardware_profile: dict[str, Any],
    stage_timings: dict[str, Any],
    environment: dict[str, Any],
    commands: list[str],
    recommended_config: dict[str, Any],
    application_record: dict[str, Any] | None = None,
) -> Path:
    root = execution_root(out_dir, execution_id)
    write_reports(
        root,
        report=report,
        hardware_profile=hardware_profile,
        stage_timings=stage_timings,
        environment=environment,
        commands=commands,
        recommended_config=recommended_config,
    )
    write_atomic(
        root / "execution_manifest.json",
        json.dumps(
            {
                "execution_id": execution_id,
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "timestamp_utc": now_utc(),
                "provenance": report_provenance_kind(report),
                "report_sha256": hashlib.sha256(
                    json.dumps(_json_safe(report), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    if application_record is not None:
        write_atomic(
            root / "application_record.json",
            json.dumps(_json_safe(application_record), indent=2, sort_keys=True),
        )
    (root / "completed.marker").write_text(now_utc(), encoding="utf-8")
    return root


def load_calibration_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_latest_completed_execution(out_dir: Path) -> Path | None:
    executions_dir = out_dir / "executions"
    if not executions_dir.exists():
        return None
    candidates = []
    for path in sorted(executions_dir.iterdir()):
        if (path / "completed.marker").exists() and (path / "calibration_report.json").exists():
            report = load_calibration_report(path / "calibration_report.json")
            if report and report_has_complete_real_evidence(report):
                candidates.append(path)
    return max(candidates, key=lambda p: p.name) if candidates else None


def recover_legacy_execution(out_dir: Path) -> Path | None:
    legacy_report = load_calibration_report(out_dir / "calibration_report.json")
    if not legacy_report or not report_has_complete_real_evidence(legacy_report):
        return None
    execution_id = legacy_report.get("execution_id") or execution_id_for_report(legacy_report)
    root = execution_root(out_dir, execution_id)
    if root.exists() and (root / "completed.marker").exists():
        return root
    hardware_profile = load_calibration_report(out_dir / "hardware_profile.json") or {}
    stage_timings = load_calibration_report(out_dir / "stage_timings.json") or {}
    environment = load_calibration_report(out_dir / "environment.json") or {}
    commands = (out_dir / "commands.txt").read_text(encoding="utf-8").splitlines()
    recommended_config_path = out_dir / "recommended_full_config.yaml"
    recommended_config = {}
    if recommended_config_path.exists():
        recommended_payload = yaml.safe_load(recommended_config_path.read_text(encoding="utf-8"))
        if isinstance(recommended_payload, dict):
            recommended_config = recommended_payload.get("experiment", recommended_payload)
    write_execution_artifacts(
        out_dir,
        execution_id=execution_id,
        report=legacy_report,
        hardware_profile=hardware_profile,
        stage_timings=stage_timings,
        environment=environment,
        commands=commands,
        recommended_config=recommended_config,
    )
    return root


@dataclass
class CalibrationState:
    model_family: str
    evaluated_batches: list[int] = field(default_factory=list)
    successful_batches: list[int] = field(default_factory=list)
    failed_batches: list[int] = field(default_factory=list)
    evidence_ready: bool = False
    selected_batch: int | None = None
    candidate_order: list[int] = field(default_factory=list)
    model_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "evaluated_batches": self.evaluated_batches,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "evidence_ready": self.evidence_ready,
            "selected_batch": self.selected_batch,
            "candidate_order": self.candidate_order,
            "model_results": _json_safe(self.model_results),
        }


def system_telemetry() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd()))
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {
                "name": props.name,
                "index": 0,
                "total_memory_bytes": int(props.total_memory),
            }
    except Exception:
        gpu = None
    return {
        "cpu_utilization_pct": psutil.cpu_percent(interval=None),
        "ram_usage_bytes": int(mem.used),
        "disk_free_bytes": int(disk.free),
        "gpu": gpu,
    }


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


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


def _gpu_utilization_pct() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        if props.major < 8:
            return None
    except Exception:
        return None
    return None


def _select_probe_batches(batch_size: int) -> int:
    return max(1, min(CALIBRATION_MAX_BATCHES, int(batch_size)))


def _build_probe_cfg(cfg: ExperimentConfig, *, batch_size: int) -> ExperimentConfig:
    return ExperimentConfig(
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
        save_json=False,
        batch_size=batch_size,
        extra=cfg.extra,
    )


def build_recommended_full_config(cfg: ExperimentConfig, *, batch_size: int) -> dict[str, Any]:
    payload = config_to_dict(cfg)
    payload["batch_size"] = batch_size
    payload["hardware"]["deterministic"] = True
    return payload


def update_run_config_batch(path: Path, *, batch_size: int) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "run" not in data:
        raise ValueError(f"invalid run config: {path}")
    data["batch_size"] = batch_size
    data["run"]["batch_size"] = batch_size
    write_atomic(path, yaml.safe_dump(data, sort_keys=False))


def _probe_single_candidate(
    *,
    cfg: ExperimentConfig,
    model_family: str,
    batch_size: int,
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    probe_dir = out_dir / "probes" / model_family / f"batch_{batch_size}"
    completed_marker = probe_dir / "completed.marker"
    if completed_marker.exists() and not force:
        report_path = probe_dir / "result.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
    probe_cfg = _build_probe_cfg(cfg, batch_size=batch_size)
    started = perf_counter()
    result = run_bounded_training_probe(probe_cfg, out_dir=probe_dir, iterations=30)
    result.setdefault("model_family", model_family)
    result.setdefault("batch_size", batch_size)
    result.setdefault("probe_elapsed_s", perf_counter() - started)
    probe_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        probe_dir / "result.json", json.dumps(_json_safe(result), indent=2, sort_keys=True)
    )
    completed_marker.write_text(now_utc(), encoding="utf-8")
    return result


def build_resume_state(
    *,
    model_family: str,
    candidate_order: list[int],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "candidate_order": candidate_order,
        "results": _json_safe(results),
        "evaluated_batches": [int(row["batch_size"]) for row in results],
        "successful_batches": [
            int(row["batch_size"]) for row in results if row["status"] == "stable"
        ],
        "failed_batches": [
            int(row["batch_size"]) for row in results if row["status"] in {"oom", "failed"}
        ],
        "selected_batch": select_largest_stable_batch(results),
        "evidence_ready": any(row["status"] == "stable" for row in results),
    }


def update_resume_state(
    *,
    model_family: str,
    candidate_order: list[int],
    results_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rows = flatten_candidate_rows(results_by_model)
    return build_resume_state(
        model_family=model_family, candidate_order=candidate_order, results=rows
    )


def write_resume_state(out_dir: Path, payload: dict[str, Any]) -> None:
    write_atomic(
        out_dir / "resume_state.json", json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    )


def flatten_candidate_rows(
    results_by_model: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {**row, "model_family": model_family}
        for model_family, rows in results_by_model.items()
        for row in rows
    ]


def select_common_stable_batch(results_by_model: dict[str, list[dict[str, Any]]]) -> int | None:
    common: list[int] = []
    all_batches = sorted(
        {int(row["batch_size"]) for rows in results_by_model.values() for row in rows}
    )
    for batch in all_batches:
        if all(
            any(row["batch_size"] == batch and row["status"] == "stable" for row in rows)
            for rows in results_by_model.values()
        ):
            common.append(batch)
    return min(common) if common else None
