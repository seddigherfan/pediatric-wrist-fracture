from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from wrist_fracture.config import ConfigError
from wrist_fracture.provenance import collect_environment_report, git_commit, git_dirty, to_jsonable
from wrist_fracture.validation_benchmark_suite import (
    MODEL_ORDER,
    benchmark_checkpoint,
    build_benchmark_image_manifest,
    discover_source_runs,
    evaluate_checkpoint,
    resolve_checkpoint_path,
    select_runs,
    validate_benchmark_image_manifest,
    validate_required_numeric_fields,
)


def _parse_models(raw: str | None) -> list[str]:
    if not raw:
        return list(MODEL_ORDER)
    models = [m.strip() for m in raw.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_ORDER:
            raise ConfigError(f"unknown model: {m}")
    return [m for m in MODEL_ORDER if m in models]


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _summary_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


@dataclass
class ModelResult:
    model_family: str
    source_smoke_run: str
    source_checkpoint: str
    source_checkpoint_sha256: str | None
    checkpoint: str
    checkpoint_sha256: str | None
    status: str
    evaluation_path: str | None = None
    benchmark_path: str | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    validation_duration_seconds: float | None = None
    preprocess_time_ms: float | None = None
    inference_time_ms: float | None = None
    postprocess_time_ms: float | None = None
    loss_time_ms: float | None = None
    mean_latency_seconds: float | None = None
    median_latency_seconds: float | None = None
    std_latency_seconds: float | None = None
    min_latency_seconds: float | None = None
    max_latency_seconds: float | None = None
    p90_latency_seconds: float | None = None
    p95_latency_seconds: float | None = None
    p99_latency_seconds: float | None = None
    throughput_images_per_second: float | None = None
    peak_allocated_vram_bytes: int | None = None
    peak_reserved_vram_bytes: int | None = None
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    flops_g: float | None = None
    checkpoint_size_bytes: int | None = None
    nms_end_to_end_mode: str | None = None
    error_summary: str | None = None


def _finish_marker(path: Path, ok: bool) -> None:
    if ok:
        path.write_text(_now(), encoding="utf-8")
    else:
        if path.exists():
            path.unlink()


def _sample_manifest(rows: list[Path]) -> list[dict[str, Any]]:
    return [{"order": i + 1, "image_path": str(path)} for i, path in enumerate(rows)]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _can_resume(output_dir: Path, marker_name: str) -> bool:
    return (output_dir / marker_name).exists()


def _load_eval_metrics(out_dir: Path) -> dict[str, Any] | None:
    payload = _load_json(out_dir / "metrics.json")
    if not payload:
        return None
    return {"metrics": payload}


def _load_benchmark_result(out_dir: Path) -> dict[str, Any] | None:
    payload = _load_json(out_dir / "benchmark.json")
    if not payload:
        return None
    return {
        "latency": payload.get("latency_seconds", {}),
        "complexity": payload.get("complexity", {}),
        "selected_images": payload.get("sample_manifest", []),
    }


def _require_valid_metrics(result: ModelResult) -> list[str]:
    payload = asdict(result)
    return validate_required_numeric_fields(
        payload,
        [
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95",
            "validation_duration_seconds",
            "mean_latency_seconds",
            "median_latency_seconds",
            "std_latency_seconds",
            "min_latency_seconds",
            "max_latency_seconds",
            "p90_latency_seconds",
            "p95_latency_seconds",
            "p99_latency_seconds",
            "throughput_images_per_second",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--suite-id")
    parser.add_argument("--models")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()

    models = _parse_models(args.models)
    source_suite = Path(args.source_suite)
    source_runs = select_runs(discover_source_runs(source_suite), models)
    suite_id = args.suite_id or f"validation-benchmark-{_now().replace(':', '').replace('-', '')}"
    suite_dir = Path("outputs/validation_benchmark_suites") / suite_id
    plan = {
        "suite_id": suite_id,
        "source_suite": str(source_suite),
        "models": models,
        "suite_dir": str(suite_dir),
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if suite_dir.exists() and not (args.resume or args.force or args.skip_completed):
        raise ConfigError(f"output collision: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t0 = perf_counter()
    env = collect_environment_report(Path.cwd())
    timings["environment"] = perf_counter() - t0

    protocol = {
        "dataset_yaml": None,
        "validation_split": "val",
        "benchmark_split": "val",
        "benchmark_batch_size": args.benchmark_batch_size,
        "warmup": args.warmup,
        "samples": args.samples,
        "io_workers": args.io_workers,
        "device_policy": "cuda:0",
        "amp_policy": "resolved from hardware config",
        "timing_policy": "CUDA synchronized around benchmarked inference regions",
        "sample_policy": "deterministic ordered manifest shared across models",
        "no_test_split": True,
        "shared_fields": [
            "dataset yaml hash",
            "validation split",
            "image size",
            "benchmark image manifest",
            "benchmark sample count",
            "benchmark batch size",
            "confidence threshold",
            "IoU threshold",
            "maximum detections",
            "device",
            "AMP/inference precision policy",
            "worker/prefetch policy",
            "warm-up count",
            "timing methodology",
        ],
        "allowed_differences": [
            "model family",
            "checkpoint",
            "architecture",
            "parameters",
            "FLOPs",
            "model size",
            "NMS/end-to-end behavior",
        ],
    }

    commands: list[str] = []
    diagnostics: list[str] = []
    sample_manifest: list[dict[str, Any]] = []
    manifest_reason: str | None = None
    rows: list[ModelResult] = []
    benchmark_sample_paths: list[Path] | None = None
    benchmark_manifest_path = suite_dir / "benchmark_image_manifest.json"
    manifest_images_root = Path("data/processed/yolo/images")
    manifest = _load_json(benchmark_manifest_path)
    if manifest is not None:
        issues = validate_benchmark_image_manifest(manifest, manifest_images_root, "val")
        if issues:
            manifest_reason = "; ".join(issues)
            manifest = None
    if manifest is None:
        manifest = build_benchmark_image_manifest(
            manifest_images_root, split="val", samples=args.samples, seed=0
        )
        manifest["invalidated_previous_manifest_reason"] = manifest_reason
        _atomic_write(benchmark_manifest_path, manifest)
    benchmark_sample_paths = [manifest_images_root / rel for rel in manifest["selected_samples"]]
    sample_manifest = _sample_manifest(benchmark_sample_paths)

    try:
        for run in source_runs:
            model_family = run["model_family"]
            resolution = resolve_checkpoint_path(
                run["checkpoint"], sha256=run.get("checkpoint_sha256")
            )
            checkpoint = Path(resolution.selected)
            diagnostics.extend(
                [
                    f"model: {model_family}",
                    f"  source: {resolution.source}",
                    f"  attempted: {list(resolution.candidates)}",
                    f"  selected: {resolution.selected}",
                ]
            )
            eval_dir = suite_dir / model_family / "evaluation"
            bench_dir = suite_dir / model_family / "benchmark"
            if args.print_commands:
                commands.append(
                    f"uv run python scripts/evaluate.py --execute --checkpoint {checkpoint}"
                )
                commands.append(
                    f"uv run python scripts/benchmark.py --execute --checkpoint {checkpoint}"
                )
            eval_result = _load_eval_metrics(eval_dir) if args.resume else None
            if eval_result is None:
                eval_result = evaluate_checkpoint(
                    checkpoint=checkpoint,
                    cfg_path=Path("configs/experiment.yaml"),
                    model_cfg=Path(f"configs/models/{model_family}.yaml"),
                    hardware_cfg=Path("configs/hardware/rtx4090.yaml"),
                    run_cfg=Path("configs/runs/smoke.yaml"),
                    split="val",
                    execute=True,
                    out_dir=eval_dir,
                )
            bench_result = _load_benchmark_result(bench_dir) if args.resume else None
            if bench_result is None:
                bench_result = benchmark_checkpoint(
                    checkpoint=checkpoint,
                    images=benchmark_sample_paths,
                    cfg_path=Path("configs/experiment.yaml"),
                    model_cfg=Path(f"configs/models/{model_family}.yaml"),
                    hardware_cfg=Path("configs/hardware/rtx4090.yaml"),
                    run_cfg=Path("configs/runs/smoke.yaml"),
                    warmup=args.warmup,
                    samples=args.samples,
                    batch_size=args.benchmark_batch_size,
                    execute=True,
                    out_dir=bench_dir,
                )
            metrics = eval_result["metrics"]
            bench_stats = bench_result["latency"]
            result = ModelResult(
                model_family=model_family,
                source_smoke_run=str(run["run_path"]),
                source_checkpoint=resolution.source,
                source_checkpoint_sha256=resolution.sha256,
                checkpoint=str(checkpoint),
                checkpoint_sha256=resolution.sha256 or metrics.get("checkpoint_sha256"),
                status="completed",
                evaluation_path=str(eval_dir),
                benchmark_path=str(bench_dir),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1=metrics.get("f1"),
                map50=metrics.get("map50"),
                map50_95=metrics.get("map50_95"),
                validation_duration_seconds=metrics.get("validation_duration_seconds"),
                preprocess_time_ms=metrics.get("preprocess_time_ms"),
                inference_time_ms=metrics.get("inference_time_ms"),
                postprocess_time_ms=metrics.get("postprocess_time_ms"),
                loss_time_ms=metrics.get("loss_time_ms"),
                mean_latency_seconds=bench_stats.get("mean"),
                median_latency_seconds=bench_stats.get("median"),
                std_latency_seconds=bench_stats.get("std"),
                min_latency_seconds=bench_stats.get("min"),
                max_latency_seconds=bench_stats.get("max"),
                p90_latency_seconds=bench_stats.get("p90"),
                p95_latency_seconds=bench_stats.get("p95"),
                p99_latency_seconds=bench_stats.get("p99"),
                throughput_images_per_second=bench_stats.get("throughput"),
                peak_allocated_vram_bytes=bench_result["complexity"].get("peak_allocated_bytes")
                if isinstance(bench_result.get("complexity"), dict)
                else None,
                peak_reserved_vram_bytes=bench_result["complexity"].get("peak_reserved_bytes")
                if isinstance(bench_result.get("complexity"), dict)
                else None,
                total_parameters=bench_result["complexity"].get("total_parameters")
                if isinstance(bench_result.get("complexity"), dict)
                else None,
                trainable_parameters=bench_result["complexity"].get("trainable_parameters")
                if isinstance(bench_result.get("complexity"), dict)
                else None,
                flops_g=bench_result["complexity"].get("flops_g")
                if isinstance(bench_result.get("complexity"), dict)
                else None,
                checkpoint_size_bytes=bench_result["complexity"].get("checkpoint_size_bytes")
                if isinstance(bench_result.get("complexity"), dict)
                else None,
                nms_end_to_end_mode=(
                    "native_end_to_end"
                    if bench_result["complexity"].get("nms_free_default")
                    else "nms_based"
                )
                if isinstance(bench_result.get("complexity"), dict)
                else None,
            )
            result.error_summary = None
            issues = _require_valid_metrics(result)
            if issues:
                raise ConfigError("; ".join(issues))
            rows.append(result)
        ok = True
    except Exception as exc:
        ok = False
        rows.append(
            ModelResult(
                model_family="unknown",
                source_smoke_run=str(source_suite),
                source_checkpoint="",
                source_checkpoint_sha256=None,
                checkpoint="",
                checkpoint_sha256=None,
                status="failed",
                error_summary=str(exc),
            )
        )
        if not args.continue_on_error:
            raise
    finally:
        if args.print_commands:
            (suite_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
        if diagnostics:
            (suite_dir / "checkpoint_resolution.txt").write_text(
                "\n".join(diagnostics) + "\n", encoding="utf-8"
            )

    summary_rows = [asdict(row) for row in rows if row.model_family != "unknown"]
    suite_summary = {
        "suite_id": suite_id,
        "source_suite": str(source_suite),
        "generated_at": _now(),
        "models": [{k: _summary_value(v) for k, v in row.items()} for row in summary_rows],
        "note": (
            "These results validate the evaluation and benchmark pipeline using one-epoch "
            "smoke checkpoints and are not final scientific model-comparison results."
        ),
    }

    _atomic_write(suite_dir / "suite_summary.json", suite_summary)
    with (suite_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted(summary_rows[0].keys()) if summary_rows else []
        )
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    report_lines = [
        "# Validation and Benchmark Suite",
        "",
        suite_summary["note"],
        "",
    ]
    for row in summary_rows:
        report_lines.extend(
            [
                f"## {row['model_family']}",
                f"- status: {row['status']}",
                f"- checkpoint: {row['checkpoint']}",
                f"- Precision: {row['precision']}",
                f"- Recall: {row['recall']}",
                f"- F1: {row['f1']}",
                f"- mAP@0.5: {row['map50']}",
                f"- mAP@0.5:0.95: {row['map50_95']}",
                f"- validation duration: {row['validation_duration_seconds']}",
                f"- mean latency: {row['mean_latency_seconds']}",
                f"- p95 latency: {row['p95_latency_seconds']}",
                f"- peak allocated VRAM: {row['peak_allocated_vram_bytes']}",
                f"- peak reserved VRAM: {row['peak_reserved_vram_bytes']}",
                "",
            ]
        )
    (suite_dir / "suite_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    _atomic_write(suite_dir / "protocol.json", protocol)
    _atomic_write(
        suite_dir / "environment.json",
        {
            "suite_id": suite_id,
            "generated_at": _now(),
            "git_commit": git_commit(Path.cwd()),
            "git_dirty": git_dirty(Path.cwd()),
            "environment": to_jsonable(env),
        },
    )
    (suite_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    if diagnostics:
        (suite_dir / "checkpoint_resolution.txt").write_text(
            "\n".join(diagnostics) + "\n", encoding="utf-8"
        )
    with (suite_dir / "benchmark_sample_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=["order", "image_path"])
        writer.writeheader()
        writer.writerows(sample_manifest)
    if manifest_reason:
        _atomic_write(
            suite_dir / "benchmark_image_manifest_invalidation.json",
            {"reason": manifest_reason, "regenerated": True},
        )
    _atomic_write(suite_dir / "stage_timings.json", timings)
    if ok:
        _finish_marker(suite_dir / "completed.marker", True)
    else:
        _finish_marker(suite_dir / "failed.marker", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
