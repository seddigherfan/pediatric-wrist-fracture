from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from scripts import benchmark, evaluate
from scripts import run_validation_benchmark_suite as suite

from wrist_fracture import runtime
from wrist_fracture.config import ConfigError
from wrist_fracture.validation_benchmark_suite import (
    _evaluation_paths,
    _percentile,
    _safe_copy_tree,
    _summary_stats,
    build_benchmark_image_manifest,
    resolve_checkpoint_path,
    select_runs,
    validate_benchmark_image_manifest,
)


def test_percentile_and_summary_stats():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.5) == 2.5
    stats = _summary_stats(values)
    assert stats["mean"] == 2.5
    assert stats["p95"] is not None
    assert stats["throughput"] == pytest.approx(1 / 2.5)


def test_select_runs_preserves_requested_order():
    runs = [
        {"model_family": "yolo26"},
        {"model_family": "yolov8"},
        {"model_family": "yolov9"},
    ]
    selected = select_runs(runs, ["yolov8", "yolov9", "yolo26"])
    assert [run["model_family"] for run in selected] == ["yolov8", "yolov9", "yolo26"]


def test_evaluate_refuses_test_without_allow_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        evaluate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/smoke.yaml",
            checkpoint=str(checkpoint),
            split="test",
            allow_test=False,
            dry_run=False,
            execute=True,
            evaluation_id=None,
            output_dir=str(tmp_path / "out"),
            preflight=False,
        ),
    )
    with pytest.raises(ConfigError, match="test evaluation requires --allow-test"):
        evaluate.main()


def test_benchmark_requires_explicit_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        benchmark.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config=None,
            hardware_config=None,
            run_config=None,
            checkpoint=str(checkpoint),
            dry_run=False,
            execute=False,
            warmup=30,
            samples=300,
            batch_size=1,
            device="cuda:0",
            benchmark_id=None,
            output_dir=str(tmp_path / "bench"),
        ),
    )
    with pytest.raises(ConfigError, match="benchmark requires explicit --execute"):
        benchmark.main()


def test_suite_dry_run_refuses_execute_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "suite_summary.json").write_text(
        json.dumps(
            {
                "models": [
                    {"model_family": "yolov8"},
                    {"model_family": "yolov9"},
                    {"model_family": "yolo26"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            source_suite=str(source),
            dry_run=True,
            execute=False,
            suite_id="suite",
            models=None,
            skip_completed=False,
            continue_on_error=False,
            resume=False,
            force=False,
            warmup=30,
            samples=300,
            benchmark_batch_size=1,
            io_workers=4,
            print_commands=False,
        ),
    )
    assert suite.main() == 0


def test_suite_missing_model_run_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "suite_summary.json").write_text(
        json.dumps({"models": [{"model_family": "yolov8"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="missing requested model runs"):
        select_runs([{"model_family": "yolov8"}], ["yolov8", "yolov9"])


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", "cpu"),
        ("cuda", "0"),
        ("cuda:0", "0"),
        ("cuda:1", "1"),
        (0, "0"),
        (1, "1"),
    ],
)
def test_normalize_device_cases(device, expected):
    assert runtime.normalize_device(device) == expected


def test_package_modules_import_without_scripts_hack():
    __import__("wrist_fracture.validation_benchmark_suite")
    __import__("wrist_fracture.smoke_suite")
    __import__("wrist_fracture.runtime")


def test_validation_benchmark_module_has_no_scripts_dependency():
    import wrist_fracture.validation_benchmark_suite as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "from scripts" not in source
    assert "import scripts" not in source


def test_resolve_checkpoint_accepts_direct_pt(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(checkpoint)
    assert resolved.selected == str(checkpoint.resolve())
    assert resolved.candidates == (str(checkpoint),)


def test_resolve_checkpoint_accepts_project_run_root(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(tmp_path)
    assert resolved.selected == str(checkpoint.resolve())


def test_resolve_checkpoint_accepts_raw_ultralytics_run_root(tmp_path: Path):
    checkpoint = tmp_path / "raw" / "train" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(tmp_path)
    assert resolved.selected == str(checkpoint.resolve())


def test_resolve_checkpoint_missing(tmp_path: Path):
    with pytest.raises(ConfigError, match="checkpoint missing or invalid"):
        resolve_checkpoint_path(tmp_path / "missing")


def test_resolve_checkpoint_rejects_zero_byte_file(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"")
    with pytest.raises(ConfigError, match="checkpoint missing or invalid"):
        resolve_checkpoint_path(checkpoint)


def test_resolve_checkpoint_does_not_append_to_existing_pt(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(checkpoint)
    assert "weights/best.pt" not in resolved.selected
    assert "best.pt/weights/best.pt" not in resolved.selected


def test_smoke_suite_model_records_preserve_checkpoint_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source"
    source.mkdir()
    run_root = tmp_path / "runs"
    run_root.mkdir()
    records = []
    for model in ["yolov8", "yolov9", "yolo26"]:
        checkpoint = tmp_path / model / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(model.encode("utf-8"))
        records.append(
            {
                "model_family": model,
                "run_path": str(run_root / model),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": f"sha-{model}",
            }
        )
    (source / "suite_summary.json").write_text(json.dumps({"models": records}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            source_suite=str(source),
            dry_run=False,
            execute=True,
            suite_id="suite",
            models=None,
            skip_completed=False,
            continue_on_error=True,
            resume=False,
            force=True,
            warmup=0,
            samples=1,
            benchmark_batch_size=1,
            io_workers=0,
            print_commands=False,
        ),
    )
    monkeypatch.setattr(
        suite,
        "evaluate_checkpoint",
        lambda **kwargs: {"metrics": {"checkpoint_sha256": kwargs["checkpoint"].name}},
    )
    monkeypatch.setattr(
        suite,
        "benchmark_checkpoint",
        lambda **kwargs: {"latency": {"mean": 1.0}, "complexity": {}},
    )
    monkeypatch.setattr(suite, "discover_source_runs", lambda _: records)
    monkeypatch.setattr(suite, "collect_environment_report", lambda _: {})
    monkeypatch.setattr(suite, "git_commit", lambda _: None)
    monkeypatch.setattr(suite, "git_dirty", lambda _: False)
    monkeypatch.setattr(suite, "to_jsonable", lambda value: value)
    monkeypatch.setattr(suite, "_finish_marker", lambda *args, **kwargs: None)
    assert suite.main() == 0
    summary = json.loads(
        (tmp_path / "outputs/validation_benchmark_suites/suite/suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["checkpoint"] for row in summary["models"]] == [
        rec["checkpoint"] for rec in records
    ]
    assert [row["checkpoint_sha256"] for row in summary["models"]] == [
        rec["checkpoint_sha256"] for rec in records
    ]


def _write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), color=color)
    img.save(path)


def test_benchmark_manifest_excludes_non_images(tmp_path: Path):
    images_root = tmp_path / "data/processed/yolo/images"
    val_root = images_root / "val"
    _write_image(val_root / "a.png")
    _write_image(val_root / "b.jpg")
    _write_image(val_root / "c.tiff")
    (val_root / "sample.npy").write_bytes(b"npy")
    (val_root / "labels.txt").write_text("label", encoding="utf-8")
    (val_root / "meta.json").write_text("{}", encoding="utf-8")
    (val_root / ".hidden.png").write_bytes(b"hidden")
    (val_root / "zero.png").write_bytes(b"")
    broken = val_root / "broken.png"
    try:
        broken.symlink_to(val_root / "missing.png")
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    manifest = build_benchmark_image_manifest(images_root, split="val", samples=10)
    assert manifest["selected_sample_count"] == 3
    assert manifest["candidate_count"] >= 8
    assert manifest["excluded_file_counts"]["unsupported_suffix"] >= 3
    assert manifest["excluded_file_counts"]["zero_byte"] == 1
    assert manifest["excluded_file_counts"]["broken_link"] == 1
    assert manifest["excluded_file_counts"]["hidden"] == 1
    assert manifest["selected_samples"] == ["val/a.png", "val/b.jpg", "val/c.tiff"]


def test_validate_benchmark_manifest_rejects_stale_entries(tmp_path: Path):
    images_root = tmp_path / "data/processed/yolo/images"
    val_root = images_root / "val"
    _write_image(val_root / "a.png")
    manifest = build_benchmark_image_manifest(images_root, split="val", samples=1)
    manifest["selected_samples"] = ["val/missing.png"]
    issues = validate_benchmark_image_manifest(manifest, images_root, "val")
    assert issues


def test_manifest_is_deterministic_and_shared_across_models(tmp_path: Path):
    images_root = tmp_path / "data/processed/yolo/images"
    val_root = images_root / "val"
    for name in ["c.png", "a.jpg", "b.tiff"]:
        _write_image(val_root / name)
    first = build_benchmark_image_manifest(images_root, split="val", samples=2, seed=0)
    second = build_benchmark_image_manifest(images_root, split="val", samples=2, seed=0)
    assert first["selected_samples"] == second["selected_samples"]
    assert first["selected_samples"] == ["val/a.jpg", "val/b.tiff"]


def test_suite_regenerates_stale_manifest_and_resolves_output_paths(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "suite_summary.json").write_text(
        json.dumps({"models": [{"model_family": "yolov8", "checkpoint": "x.pt", "run_path": "r"}]}),
        encoding="utf-8",
    )
    images_root = tmp_path / "data/processed/yolo/images/val"
    _write_image(images_root / "a.png")
    monkeypatch.chdir(tmp_path)
    suite_dir = tmp_path / "outputs/validation_benchmark_suites/suite"
    suite_dir.mkdir(parents=True)
    (suite_dir / "benchmark_image_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "val",
                "images_root": str((tmp_path / "data/processed/yolo/images").resolve()),
                "allowed_image_suffixes": [".png"],
                "selected_samples": ["val/missing.png"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            source_suite=str(source),
            dry_run=False,
            execute=True,
            suite_id="suite",
            models="yolov8",
            skip_completed=True,
            continue_on_error=True,
            resume=True,
            force=False,
            warmup=0,
            samples=1,
            benchmark_batch_size=1,
            io_workers=0,
            print_commands=False,
        ),
    )
    monkeypatch.setattr(
        suite,
        "discover_source_runs",
        lambda _: [{"model_family": "yolov8", "checkpoint": "x.pt", "run_path": "r"}],
    )
    monkeypatch.setattr(
        suite,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: SimpleNamespace(
            source="x.pt",
            candidates=("x.pt",),
            selected=str(tmp_path / "x.pt"),
            sha256=None,
        ),
    )
    (tmp_path / "x.pt").write_bytes(b"model")
    monkeypatch.setattr(
        suite,
        "evaluate_checkpoint",
        lambda **kwargs: {
            "metrics": {
                "precision": 1,
                "recall": 1,
                "f1": 1,
                "map50": 1,
                "map50_95": 1,
                "validation_duration_seconds": 1,
                "preprocess_time_ms": 1,
                "inference_time_ms": 1,
                "postprocess_time_ms": 1,
                "loss_time_ms": 1,
            }
        },
    )
    monkeypatch.setattr(
        suite,
        "benchmark_checkpoint",
        lambda **kwargs: {
            "latency": {
                "mean": 1,
                "median": 1,
                "std": 0,
                "min": 1,
                "max": 1,
                "p90": 1,
                "p95": 1,
                "p99": 1,
                "throughput": 1,
            },
            "complexity": {},
        },
    )
    monkeypatch.setattr(suite, "collect_environment_report", lambda _: {})
    monkeypatch.setattr(suite, "git_commit", lambda _: None)
    monkeypatch.setattr(suite, "git_dirty", lambda _: False)
    monkeypatch.setattr(suite, "to_jsonable", lambda value: value)
    assert suite.main() == 0
    regenerated = json.loads(
        (suite_dir / "benchmark_image_manifest.json").read_text(encoding="utf-8")
    )
    assert regenerated["selected_sample_count"] == 1
    assert (tmp_path / "outputs/validation_benchmark_suites/suite/suite_summary.json").exists()


def test_safe_copy_tree_handles_overlap_and_merge(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    (src / "a" / "file.txt").write_text("one", encoding="utf-8")
    _safe_copy_tree(src, dst)
    assert (dst / "a" / "file.txt").read_text(encoding="utf-8") == "one"
    _safe_copy_tree(src, src)
    with pytest.raises(ConfigError):
        _safe_copy_tree(src, src / "child")
    with pytest.raises(ConfigError):
        _safe_copy_tree(src / "a", src)
    other = tmp_path / "other"
    (other / "b").mkdir(parents=True)
    (other / "b" / "file.txt").write_text("two", encoding="utf-8")
    _safe_copy_tree(other, dst)
    assert (dst / "b" / "file.txt").read_text(encoding="utf-8") == "two"


def test_evaluate_checkpoint_recovers_existing_framework_output(tmp_path: Path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import wrist_fracture.validation_benchmark_suite as vbs

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def val(self, **kwargs):
            return SimpleNamespace(
                results_dict={
                    "metrics/precision(B)": 1,
                    "metrics/recall(B)": 1,
                    "metrics/mAP50(B)": 1,
                    "metrics/mAP50-95(B)": 1,
                    "nt_per_class": [1],
                },
                speed={"preprocess": 1, "inference": 1, "loss": 1, "postprocess": 1},
                maps=[1],
                fitness=1,
                files=[],
            )

    out_dir = tmp_path / "evaluation"
    paths = _evaluation_paths(out_dir)
    (paths["framework_run_dir"]).mkdir(parents=True, exist_ok=True)
    (paths["framework_run_dir"] / "predictions.json").write_text("{}", encoding="utf-8")
    (paths["framework_run_dir"] / "val_batch0_pred.jpg").write_bytes(b"img")
    metrics_payload = {"precision": 1, "recall": 1, "validation_duration_seconds": 1}
    (paths["metrics_path"]).write_text(json.dumps(metrics_payload), encoding="utf-8")
    monkeypatch.setattr(
        vbs,
        "load_config_bundle",
        lambda *args, **kwargs: SimpleNamespace(
            dataset_yaml=tmp_path / "dataset.yaml",
            image_size=320,
            batch_size=1,
            seed=42,
            model=SimpleNamespace(family="yolov9"),
            hardware=SimpleNamespace(workers=0, device="cpu", amp=False),
        ),
    )
    monkeypatch.setattr(vbs, "validate_experiment_config", lambda cfg, dry_run=False: [])
    monkeypatch.setattr(vbs, "collect_environment_report", lambda _: {})
    monkeypatch.setattr(vbs, "git_commit", lambda _: None)
    monkeypatch.setattr(vbs, "git_dirty", lambda _: False)
    monkeypatch.setattr(vbs, "sha256_file", lambda path: "sha")
    monkeypatch.setattr(vbs, "to_jsonable", lambda value: value)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO, __version__="0"))
    result = vbs.evaluate_checkpoint(
        checkpoint=tmp_path / "yolov9.pt",
        cfg_path=tmp_path / "cfg.yaml",
        model_cfg=None,
        hardware_cfg=None,
        run_cfg=None,
        split="val",
        execute=True,
        out_dir=out_dir,
    )
    assert result["metrics"]["precision"] == 1
    assert (out_dir / "raw" / "predictions.json").exists()
    assert not (out_dir / "raw" / "raw").exists()
    assert not (out_dir / "raw" / "val" / "raw").exists()
    assert (out_dir / "completed.marker").exists()
