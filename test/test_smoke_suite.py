from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from wrist_fracture import smoke_suite


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "configs/models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/hardware").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/yolo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/yolo/dataset.yaml").write_text(
        "path: data\ntrain: train\nval: val\ntest: test\nnames: {0: fracture}\n",
        encoding="utf-8",
    )
    (tmp_path / "configs/experiment.yaml").write_text(
        (
            "experiment:\n"
            "  dataset:\n"
            "    yaml: data/processed/yolo/dataset.yaml\n"
            "  model:\n"
            "    family: yolov8\n"
            "    checkpoint: yolov8n.pt\n"
            "    scale: n\n"
            "  hardware:\n"
            "    device: cuda:0\n"
            "    amp: true\n"
            "    workers: 0\n"
            "    cache: ram\n"
            "    deterministic: true\n"
            "    allow_cpu_training: false\n"
            "    require_gpu: true\n"
            "  run:\n"
            "    name: smoke\n"
            "    output_root: outputs/experiments\n"
            "    resume: false\n"
            "    save_period: 1\n"
            "    validation_split: val\n"
            "    test_split: test\n"
            "    allow_test_evaluation: false\n"
            "    selection_metric: metrics/mAP50-95(B)\n"
            "    repeated_runs: 1\n"
            "    batch_size_policy: fixed\n"
            "  image_size: 320\n"
            "  epochs: 1\n"
            "  patience: 1\n"
            "  seed: 42\n"
            "  pretrained: true\n"
            "  optimizer: auto\n"
            "  lr0: 0.01\n"
            "  lrf: 0.01\n"
            "  weight_decay: 0.0005\n"
            "  augmentation: {}\n"
            "  batch_size: 4\n"
        ),
        encoding="utf-8",
    )
    for name, checkpoint in [
        ("yolov8", "yolov8n.pt"),
        ("yolov9", "yolov9t.pt"),
        ("yolo26", "yolo26n.pt"),
    ]:
        (tmp_path / f"configs/models/{name}.yaml").write_text(
            f"model:\n  family: {name}\n  checkpoint: {checkpoint}\n  scale: n\n",
            encoding="utf-8",
        )
    (tmp_path / "configs/hardware/rtx4090.yaml").write_text(
        (
            "hardware:\n"
            "  device: cuda:0\n"
            "  amp: true\n"
            "  workers: 8\n"
            "  cache: disk\n"
            "  deterministic: false\n"
            "  allow_cpu_training: false\n"
            "  require_gpu: true\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "configs/runs/smoke.yaml").write_text(
        (
            "run:\n"
            "  name: smoke\n"
            "  output_root: outputs/experiments\n"
            "  resume: false\n"
            "  save_period: 1\n"
            "  validation_split: val\n"
            "  test_split: test\n"
            "  allow_test_evaluation: false\n"
            "  selection_metric: metrics/mAP50-95(B)\n"
            "  repeated_runs: 1\n"
            "  batch_size_policy: fixed\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "outputs/dataset_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs/dataset_reports/final_dataset_audit.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "data/splits").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (tmp_path / f"data/splits/{split}_patients.txt").write_text("1\n", encoding="utf-8")


def test_suite_planning_and_model_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    plan = smoke_suite.normalize_models("yolo26,yolov8,yolov9")
    assert plan == ["yolov8", "yolov9", "yolo26"]


def test_refuses_without_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        smoke_suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=True,
            execute=False,
            suite_id="smoke-test",
            models=None,
            continue_on_error=False,
            recover=False,
            skip_completed=False,
            print_commands=False,
        ),
    )
    assert smoke_suite.main() == 0
    out = capsys.readouterr().out
    assert "suite_id" in out


def test_smoke_caps_enforced():
    cfg = smoke_suite.build_config("yolov8", "suite", root=Path("."))
    cfg = cfg.__class__(**{**cfg.__dict__, "epochs": 2})
    assert smoke_suite.validate_smoke_caps(cfg)


def test_unique_run_ids():
    assert smoke_suite.build_config("yolov8", "x", root=Path(".")).run.run_id == "yolov8/x"


def test_artifact_validation_detects_missing(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    assert smoke_suite.validate_artifacts(root)


def test_compare_configs_reports_differences():
    diffs = smoke_suite.compare_configs(
        [
            {
                "model_family": "yolov8",
                "dataset_yaml_sha256": "a",
                "train_split": "train",
                "validation_split": "val",
                "epochs": 1,
                "image_size": 320,
                "batch_size": 4,
                "seed": 42,
                "patience": 1,
                "optimizer": "auto",
                "lr0": 0.01,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "augmentation": {},
                "workers": 8,
                "amp": True,
                "cache": "disk",
            },
            {
                "model_family": "yolov9",
                "dataset_yaml_sha256": "b",
                "train_split": "train",
                "validation_split": "val",
                "epochs": 1,
                "image_size": 320,
                "batch_size": 4,
                "seed": 42,
                "patience": 1,
                "optimizer": "auto",
                "lr0": 0.01,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "augmentation": {},
                "workers": 8,
                "amp": True,
                "cache": "disk",
            },
        ]
    )
    assert diffs


def test_suite_report_generation(tmp_path: Path):
    suite = tmp_path / "suite"
    smoke_suite.build_reports(
        suite,
        [
            {
                "model_family": "yolov8",
                "status": "completed",
                "run_id": "a",
                "run_path": "/r",
                "checkpoint": "/c",
                "start_time": None,
                "end_time": None,
                "duration_seconds": None,
                "gpu": None,
                "peak_vram_gb": None,
                "params_m": None,
                "flops_b": None,
                "checkpoint_size_mb": None,
                "precision": 0.1,
                "recall": 0.2,
                "f1": 0.3,
                "map50": 0.4,
                "map50_95": 0.5,
                "best_epoch": 1,
                "recovery_used": False,
                "error_summary": None,
            }
        ],
    )
    assert (suite / "suite_summary.json").exists()
    assert (suite / "suite_summary.csv").exists()
    assert (suite / "suite_report.md").exists()
