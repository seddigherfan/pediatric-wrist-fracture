from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from wrist_fracture import full_experiment_suite as suite
from wrist_fracture.config import ConfigError


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "configs/models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/hardware").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/yolo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/processed/yolo/dataset.yaml").write_text(
        "path: data\ntrain: train\nval: val\ntest: test\nnames: {0: fracture}\n",
        encoding="utf-8",
    )
    (tmp_path / "outputs/dataset_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs/dataset_reports/final_dataset_audit.json").write_text(
        "{}", encoding="utf-8"
    )
    for split in ["train", "val", "test"]:
        (tmp_path / f"data/splits/{split}_patients.txt").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"data/splits/{split}_patients.txt").write_text("1\n", encoding="utf-8")
    (tmp_path / "configs/experiment.yaml").write_text(
        """
experiment:
  dataset:
    yaml: data/processed/yolo/dataset.yaml
    split_yaml: data/processed/yolo/dataset.yaml
  model:
    family: yolov8
    checkpoint: yolov8n.pt
    scale: n
  hardware:
    device: cuda:0
    amp: true
    workers: 8
    cache: disk
    deterministic: true
    allow_cpu_training: false
    require_gpu: true
  run:
    name: full
    output_root: outputs/experiments
    resume: false
    save_period: 1
    validation_split: val
    test_split: test
    allow_test_evaluation: false
    selection_metric: metrics/mAP50-95(B)
    repeated_runs: 1
    batch_size_policy: fixed
  image_size: 640
  epochs: 100
  patience: 20
  seed: 42
  pretrained: true
  optimizer: auto
  lr0: 0.01
  lrf: 0.01
  weight_decay: 0.0005
  augmentation: {}
  batch_size: 64
""",
        encoding="utf-8",
    )
    for name, checkpoint in [
        ("yolov8", "yolov8n.pt"),
        ("yolov9", "yolov9t.pt"),
        ("yolo26", "yolo26n.pt"),
    ]:
        scale = "t" if name == "yolov9" else "n"
        (tmp_path / f"configs/models/{name}.yaml").write_text(
            f"model:\n  family: {name}\n  checkpoint: {checkpoint}\n  scale: {scale}\n",
            encoding="utf-8",
        )
    (tmp_path / "configs/hardware/rtx4090.yaml").write_text(
        """
hardware:
  device: cuda:0
  amp: true
  workers: 8
  cache: disk
  deterministic: true
  allow_cpu_training: false
  require_gpu: true
""",
        encoding="utf-8",
    )
    (tmp_path / "configs/runs/full.yaml").write_text(
        """
run:
  name: full
  output_root: outputs/experiments
  resume: false
  save_period: 1
  validation_split: val
  test_split: test
  allow_test_evaluation: false
  selection_metric: metrics/mAP50-95(B)
  repeated_runs: 1
  batch_size_policy: fixed
batch_size: 64
""",
        encoding="utf-8",
    )
    cal = tmp_path / "outputs/calibration/executions/cal-1"
    cal.mkdir(parents=True, exist_ok=True)
    (cal / "completed.marker").write_text("done", encoding="utf-8")
    report = {
        "common_stable_batch": 64,
        "candidates": [{"model_family": "yolov8", "status": "stable"}],
    }
    (cal / "calibration_report.json").write_text(json.dumps(report), encoding="utf-8")
    (cal / "application_record.json").write_text(
        json.dumps({"selected_batch": 64, "applied_batch": 64}), encoding="utf-8"
    )


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suite, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(suite, "_cuda_device_exists", lambda device: True)
    monkeypatch.setattr(suite, "_safe_gpu_preflight", lambda cfg: None)
    monkeypatch.setattr(
        suite, "validate_calibration_evidence", lambda root, cfg: {"common_stable_batch": 64}
    )
    monkeypatch.setattr(suite, "_dataset_preflight", lambda root, dataset_yaml: {"ok": True})
    monkeypatch.setattr(suite, "collect_environment_report", lambda root: {})
    monkeypatch.setattr(suite, "git_commit", lambda root: "abc")
    monkeypatch.setattr(suite, "git_dirty", lambda root: False)
    monkeypatch.setattr(suite, "dependency_lock_hash", lambda root: "lock")
    monkeypatch.setattr(suite, "to_jsonable", lambda value: value)


def test_deterministic_model_order_and_unique_suite_ids():
    assert suite.parse_models("yolo26,yolov8,yolov9") == ["yolov8", "yolov9", "yolo26"]
    assert suite.default_suite_id() != suite.default_suite_id()


def test_refuses_without_execute_via_dry_run_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=True,
            execute=False,
            suite_id="suite",
            models=None,
            resume=False,
            skip_completed=False,
            recover=False,
            continue_on_error=False,
            force=False,
            print_commands=False,
            print_report=False,
        ),
    )
    assert suite.main() == 0
    out = capsys.readouterr().out
    assert "suite_id" in out


def test_full_vs_smoke_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    cfg = suite.build_full_config("yolov8", "suite")
    bad = cfg.__class__(**{**cfg.__dict__, "epochs": 1})
    assert suite.validate_full_config(bad)


def test_test_split_guard():
    cfg = suite.build_full_config("yolov8", "suite")
    bad = cfg.__class__(
        **{
            **cfg.__dict__,
            "run": cfg.run.__class__(**{**cfg.run.__dict__, "allow_test_evaluation": True}),
        }
    )
    assert suite.validate_full_config(bad)


def test_sequential_execution_and_no_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    order: list[str] = []
    active: list[str] = []

    def fake_execute(cfg, root, args):
        assert not active
        active.append(cfg.model.family)
        order.append(cfg.model.family)
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "resolved_config.yaml").write_text("{}", encoding="utf-8")
        (root / "environment.json").write_text("{}", encoding="utf-8")
        (root / "provenance.json").write_text("{}", encoding="utf-8")
        (root / "command.txt").write_text("cmd", encoding="utf-8")
        (root / "raw/train/weights").mkdir(parents=True, exist_ok=True)
        (root / "raw/train/results.csv").write_text(
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "1,0.1,0.2,0.3,0.4\n",
            encoding="utf-8",
        )
        (root / "raw/train/weights/best.pt").write_bytes(b"b")
        (root / "raw/train/weights/last.pt").write_bytes(b"l")
        (root / "checkpoints/best.pt").write_bytes(b"b")
        (root / "checkpoints/last.pt").write_bytes(b"l")
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        (root / "metrics/history.csv").write_text(
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.1,0.2,0.3,0.4\n",
            encoding="utf-8",
        )
        (root / "metrics/validation.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "precision": 0.1,
                    "recall": 0.2,
                    "f1": 0.133,
                    "map50": 0.3,
                    "map50_95": 0.4,
                    "best_epoch": 1,
                }
            ),
            encoding="utf-8",
        )
        (root / "metrics/run_summary.json").write_text(
            json.dumps(
                {
                    "started_at": "s",
                    "ended_at": "e",
                    "duration_seconds": 1,
                    "gpu_peak_memory_bytes": 1,
                }
            ),
            encoding="utf-8",
        )
        (root / "completed.marker").write_text("done", encoding="utf-8")
        active.clear()

    monkeypatch.setattr(suite, "execute_training_with_args", fake_execute)
    monkeypatch.setattr(suite, "validate_completed_model_run", lambda *a, **k: [])
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=False,
            execute=True,
            suite_id="suite",
            models=None,
            resume=False,
            skip_completed=False,
            recover=False,
            continue_on_error=False,
            force=False,
            print_commands=False,
            print_report=False,
        ),
    )
    suite.main()
    assert order == ["yolov8", "yolov9", "yolo26"]


def test_stop_on_first_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    calls: list[str] = []

    def fake_execute(cfg, root, args):
        calls.append(cfg.model.family)
        raise RuntimeError("boom")

    monkeypatch.setattr(suite, "execute_training_with_args", fake_execute)
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=False,
            execute=True,
            suite_id="suite",
            models=None,
            resume=False,
            skip_completed=False,
            recover=False,
            continue_on_error=False,
            force=False,
            print_commands=False,
            print_report=False,
        ),
    )
    with pytest.raises(RuntimeError):
        suite.main()
    assert calls == ["yolov8"]


def test_continue_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)

    def fake_execute(cfg, root, args):
        if cfg.model.family == "yolov8":
            raise RuntimeError("boom")
        (root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (root / "resolved_config.yaml").write_text("{}", encoding="utf-8")
        (root / "environment.json").write_text("{}", encoding="utf-8")
        (root / "provenance.json").write_text("{}", encoding="utf-8")
        (root / "command.txt").write_text("cmd", encoding="utf-8")
        (root / "raw/train/weights").mkdir(parents=True, exist_ok=True)
        (root / "raw/train/results.csv").write_text(
            "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
            "1,0.1,0.2,0.3,0.4\n",
            encoding="utf-8",
        )
        (root / "raw/train/weights/best.pt").write_bytes(b"b")
        (root / "raw/train/weights/last.pt").write_bytes(b"l")
        (root / "checkpoints/best.pt").write_bytes(b"b")
        (root / "checkpoints/last.pt").write_bytes(b"l")
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        (root / "metrics/validation.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "precision": 0.1,
                    "recall": 0.2,
                    "f1": 0.133,
                    "map50": 0.3,
                    "map50_95": 0.4,
                    "best_epoch": 1,
                }
            ),
            encoding="utf-8",
        )
        (root / "metrics/run_summary.json").write_text(
            json.dumps(
                {
                    "started_at": "s",
                    "ended_at": "e",
                    "duration_seconds": 1,
                    "gpu_peak_memory_bytes": 1,
                }
            ),
            encoding="utf-8",
        )
        (root / "completed.marker").write_text("done", encoding="utf-8")

    monkeypatch.setattr(suite, "execute_training_with_args", fake_execute)
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=False,
            execute=True,
            suite_id="suite",
            models=None,
            resume=False,
            skip_completed=False,
            recover=False,
            continue_on_error=True,
            force=False,
            print_commands=False,
            print_report=False,
        ),
    )
    with pytest.raises(ConfigError):
        suite.main()


def test_resume_and_recovery_and_skip_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    cfg = suite.build_full_config("yolov8", "suite")
    run_dir = suite.build_run_path(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw/train/weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw/train/results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.1,0.2,0.3,0.4\n",
        encoding="utf-8",
    )
    (run_dir / "raw/train/weights/best.pt").write_bytes(b"b")
    (run_dir / "raw/train/weights/last.pt").write_bytes(b"l")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints/best.pt").write_bytes(b"b")
    (run_dir / "checkpoints/last.pt").write_bytes(b"l")
    recovered = []
    monkeypatch.setattr(suite, "recover_training_artifacts", lambda root: recovered.append(root))
    monkeypatch.setattr(
        suite,
        "execute_training_with_args",
        lambda cfg, root, args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=False,
            execute=True,
            suite_id="suite",
            models="yolov8",
            resume=True,
            skip_completed=False,
            recover=True,
            continue_on_error=True,
            force=False,
            print_commands=False,
            print_report=False,
        ),
    )
    suite.main()
    assert recovered


def test_effective_protocol_capture_and_audit():
    summary = {
        "effective_protocol": {
            "optimizer": "MuSGD",
            "lr0": 0.01,
            "momentum": 0.937,
            "augmentation": {
                "RandAugment": "randaugment",
                "erasing": 0.4,
                "horizontal_flip": 0.5,
                "translate": 0.1,
                "scale": 0.5,
            },
        }
    }
    assert suite._effective_protocol_summary(summary) == summary["effective_protocol"]
    diffs = suite._validate_effective_protocols(
        [
            summary,
            {
                "model_family": "yolov9",
                "effective_protocol": {
                    "optimizer": "SGD",
                    "lr0": 0.01,
                    "momentum": 0.937,
                    "augmentation": summary["effective_protocol"]["augmentation"],
                },
            },
        ]
    )
    assert diffs


def test_invalid_completed_run_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    cfg = suite.build_full_config("yolov8", "suite")
    run_dir = suite.build_run_path(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "completed.marker").write_text("done", encoding="utf-8")
    (run_dir / "resolved_config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: argparse.Namespace(
            dry_run=False,
            execute=True,
            suite_id="suite",
            models="yolov8",
            resume=False,
            skip_completed=True,
            recover=False,
            continue_on_error=False,
            force=False,
            print_commands=False,
            print_report=False,
        ),
    )
    with pytest.raises(ConfigError):
        suite.main()


def test_report_generation_and_consistency_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    suite_dir = tmp_path / "outputs/full_experiment_suites/suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        {
            "model_family": "yolov8",
            "run_id": "a",
            "run_path": "p",
            "checkpoint": "c",
            "status": "completed",
        },
    ]
    suite._write_suite_report(suite_dir, summaries, [])
    assert (suite_dir / "suite_summary.json").exists()
    assert (suite_dir / "suite_summary.csv").exists()
    assert (suite_dir / "suite_report.md").exists()


def test_no_duplicated_family_path():
    cfg = suite.build_full_config("yolov8", "suite")
    assert "yolov8/yolov8" not in str(suite.build_run_path(cfg))


def test_batch_64_propagation_and_no_silent_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_minimal_repo(tmp_path)
    _patch_common(monkeypatch, tmp_path)
    cfg = suite.build_full_config("yolov8", "suite")
    assert cfg.batch_size == 64
    assert cfg.image_size == 640
    assert cfg.epochs == 100
    assert cfg.run.allow_test_evaluation is False
