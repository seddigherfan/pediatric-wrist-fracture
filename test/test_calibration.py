from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import calibrate_rtx4090 as calibrate

import wrist_fracture.calibration_probe as calibration_probe
from wrist_fracture.calibration import (
    build_recommended_full_config,
    build_resume_state,
    candidate_batches,
    report_has_complete_real_evidence,
    select_largest_common_stable_batch,
    select_largest_stable_batch,
    update_run_config_batch,
    write_resume_state,
)
from wrist_fracture.config import load_config_bundle


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
            "    workers: 8\n"
            "    cache: disk\n"
            "    deterministic: false\n"
            "    allow_cpu_training: false\n"
            "    require_gpu: true\n"
            "  run:\n"
            "    name: full\n"
            "    output_root: outputs/experiments\n"
            "    resume: false\n"
            "    save_period: 1\n"
            "    validation_split: val\n"
            "    test_split: test\n"
            "    allow_test_evaluation: false\n"
            "    selection_metric: metrics/mAP50-95(B)\n"
            "    repeated_runs: 1\n"
            "    batch_size_policy: fixed\n"
            "  image_size: 640\n"
            "  epochs: 100\n"
            "  patience: 20\n"
            "  seed: 42\n"
            "  pretrained: true\n"
            "  optimizer: auto\n"
            "  lr0: 0.01\n"
            "  lrf: 0.01\n"
            "  weight_decay: 0.0005\n"
            "  augmentation: {}\n"
            "  batch_size: 1\n"
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
    (tmp_path / "configs/runs/full.yaml").write_text(
        (
            "run:\n"
            "  name: full\n"
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


def test_candidate_planning_is_bounded():
    assert candidate_batches() == [8, 16, 32, 48, 64]
    assert candidate_batches(candidates=[64, 16, 16, 8]) == [8, 16, 64]


def test_largest_stable_batch_and_common_selection():
    rows = [
        {"batch_size": 8, "status": "stable"},
        {"batch_size": 16, "status": "stable"},
        {"batch_size": 32, "status": "failed"},
    ]
    assert select_largest_stable_batch(rows) == 16
    assert (
        select_largest_common_stable_batch({"yolov8": rows, "yolov9": rows, "yolo26": rows}) == 16
    )


def test_resume_state_and_writeback(tmp_path: Path):
    payload = build_resume_state(
        model_family="yolov8",
        candidate_order=[8, 16, 32],
        results=[
            {"batch_size": 8, "status": "stable"},
            {"batch_size": 16, "status": "oom"},
            {"batch_size": 32, "status": "failed"},
        ],
    )
    write_resume_state(tmp_path, payload)
    saved = json.loads((tmp_path / "resume_state.json").read_text(encoding="utf-8"))
    assert saved["selected_batch"] == 8
    assert saved["failed_batches"] == [16, 32]


def test_config_update_and_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    update_run_config_batch(tmp_path / "configs/runs/full.yaml", batch_size=32)
    cfg = load_config_bundle(
        tmp_path / "configs/experiment.yaml",
        model_path=tmp_path / "configs/models/yolov8.yaml",
        hardware_path=tmp_path / "configs/hardware/rtx4090.yaml",
        run_path=tmp_path / "configs/runs/full.yaml",
    )
    assert cfg.batch_size == 32


def test_recommended_config_preserves_experiment_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config_bundle(
        tmp_path / "configs/experiment.yaml",
        model_path=tmp_path / "configs/models/yolov8.yaml",
        hardware_path=tmp_path / "configs/hardware/rtx4090.yaml",
        run_path=tmp_path / "configs/runs/full.yaml",
    )
    payload = build_recommended_full_config(cfg, batch_size=16)
    assert payload["batch_size"] == 16
    assert payload["epochs"] == cfg.epochs
    assert payload["hardware"]["deterministic"] is True


def test_report_generation_writes_expected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        calibration_probe,
        "run_bounded_training_probe",
        lambda cfg, **kwargs: {
            "model_family": cfg.model.family,
            "batch_size": cfg.batch_size,
            "status": "stable",
            "cuda_oom": False,
            "peak_allocated_vram_bytes": 1024,
            "peak_reserved_vram_bytes": 2048,
            "gpu_utilization": {"status": "unavailable"},
            "images_per_sec": 12.0,
            "batches_per_sec": 6.0,
            "elapsed_s": 0.5,
            "estimated_epoch_duration_s": 0.5,
            "cpu_utilization_pct": 0.0,
            "ram_usage_bytes": 1,
            "exception": None,
            "actual_optimizer_iterations": 30,
            "warmup_iterations": 5,
            "warmup_elapsed_s": 0.1,
            "iteration_mean_latency_s": 0.1,
            "iteration_median_latency_s": 0.1,
            "iteration_p95_latency_s": 0.1,
            "cache_key": "x",
            "cache_key_components": {},
            "config": {},
        },
    )
    monkeypatch.setattr(
        calibrate, "run_bounded_training_probe", calibration_probe.run_bounded_training_probe
    )
    monkeypatch.setattr(
        calibrate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/full.yaml",
            output_dir=str(tmp_path / "outputs/calibration"),
            dry_run=False,
            execute=True,
            apply=False,
            resume=False,
            force=False,
            continue_on_error=False,
            print_report=False,
            candidate_batches="8,16",
        ),
    )
    assert calibrate.main() == 0
    out_dir = tmp_path / "outputs/calibration/executions"
    exec_dirs = list(out_dir.iterdir())
    assert exec_dirs
    out_dir = exec_dirs[0]
    for name in [
        "calibration_report.json",
        "calibration_report.csv",
        "hardware_profile.json",
        "recommended_full_config.yaml",
        "stage_timings.json",
        "environment.json",
        "commands.txt",
        "resume_state.json",
    ]:
        assert (out_dir / name).exists()
    assert report_has_complete_real_evidence(
        json.loads((out_dir / "calibration_report.json").read_text(encoding="utf-8"))
    )


def test_oom_handling_and_candidate_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    seen: list[int] = []

    def fake_probe(cfg, **kwargs):
        batch = cfg.batch_size
        seen.append(batch)
        if batch == 8:
            return {
                "model_family": cfg.model.family,
                "batch_size": batch,
                "status": "oom",
                "cuda_oom": True,
                "peak_allocated_vram_bytes": 4096,
                "peak_reserved_vram_bytes": 8192,
                "gpu_utilization_pct": None,
                "images_per_sec": None,
                "epoch_duration_s": None,
                "throughput": None,
                "total_runtime_s": 0.1,
                "exception": "CUDA out of memory",
                "status_detail": "CUDA out of memory",
            }
        return {
            "model_family": cfg.model.family,
            "batch_size": batch,
            "status": "stable",
            "cuda_oom": False,
            "peak_allocated_vram_bytes": 1024,
            "peak_reserved_vram_bytes": 2048,
            "gpu_utilization_pct": None,
            "images_per_sec": 10.0,
            "epoch_duration_s": 0.5,
            "throughput": 10.0,
            "total_runtime_s": 0.5,
            "exception": None,
            "status_detail": None,
        }

    monkeypatch.setattr(calibrate, "run_bounded_training_probe", fake_probe)
    monkeypatch.setattr(calibration_probe, "run_bounded_training_probe", fake_probe)
    monkeypatch.setattr(
        calibrate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/full.yaml",
            output_dir=str(tmp_path / "outputs/calibration"),
            dry_run=False,
            execute=True,
            apply=False,
            resume=False,
            force=False,
            continue_on_error=False,
            print_report=False,
            candidate_batches="8,16",
        ),
    )
    assert calibrate.main() == 0
    assert seen == [8]
    exec_dir = next((tmp_path / "outputs/calibration/executions").iterdir())
    report = json.loads((exec_dir / "calibration_report.json").read_text(encoding="utf-8"))
    assert report["candidates"][0]["status"] == "oom"
    assert report["recommended_batch_size"] is None


def test_resume_skips_completed_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls: list[int] = []

    def fake_probe(cfg, **kwargs):
        calls.append(cfg.batch_size)
        return {
            "model_family": cfg.model.family,
            "batch_size": cfg.batch_size,
            "status": "stable",
            "cuda_oom": False,
            "peak_allocated_vram_bytes": 1024,
            "peak_reserved_vram_bytes": 2048,
            "gpu_utilization_pct": None,
            "images_per_sec": 10.0,
            "epoch_duration_s": 0.5,
            "throughput": 10.0,
            "total_runtime_s": 0.5,
            "exception": None,
            "status_detail": None,
        }

    monkeypatch.setattr(calibrate, "run_bounded_training_probe", fake_probe)
    monkeypatch.setattr(calibration_probe, "run_bounded_training_probe", fake_probe)
    monkeypatch.setattr(
        calibrate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/full.yaml",
            output_dir=str(tmp_path / "outputs/calibration"),
            dry_run=False,
            execute=True,
            apply=False,
            resume=False,
            force=False,
            continue_on_error=False,
            print_report=False,
            candidate_batches="8,16",
        ),
    )
    assert calibrate.main() == 0

    monkeypatch.setattr(
        calibrate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/full.yaml",
            output_dir=str(tmp_path / "outputs/calibration"),
            dry_run=False,
            execute=True,
            apply=False,
            resume=True,
            force=False,
            continue_on_error=False,
            print_report=False,
            candidate_batches="8,16",
        ),
    )
    assert calibrate.main() == 0
    assert calls == [8, 16]


def test_apply_updates_only_full_run_and_validates_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        calibrate,
        "run_bounded_training_probe",
        lambda cfg, **kwargs: {
            "model_family": cfg.model.family,
            "batch_size": cfg.batch_size,
            "status": "stable",
            "cuda_oom": False,
            "peak_allocated_vram_bytes": 1024,
            "peak_reserved_vram_bytes": 2048,
            "gpu_utilization": {"status": "unavailable"},
            "images_per_sec": 10.0,
            "batches_per_sec": 5.0,
            "elapsed_s": 0.5,
            "estimated_epoch_duration_s": 0.5,
            "cpu_utilization_pct": 0.0,
            "ram_usage_bytes": 1,
            "exception": None,
            "actual_optimizer_iterations": 30,
            "warmup_iterations": 5,
            "warmup_elapsed_s": 0.1,
            "iteration_mean_latency_s": 0.1,
            "iteration_median_latency_s": 0.1,
            "iteration_p95_latency_s": 0.1,
            "cache_key": "x",
            "cache_key_components": {},
            "config": {},
        },
    )
    monkeypatch.setattr(
        calibration_probe, "run_bounded_training_probe", calibrate.run_bounded_training_probe
    )
    monkeypatch.setattr(
        calibrate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/full.yaml",
            output_dir=str(tmp_path / "outputs/calibration"),
            dry_run=False,
            execute=True,
            apply=True,
            resume=False,
            force=False,
            continue_on_error=False,
            print_report=False,
            candidate_batches="8,16",
        ),
    )
    assert calibrate.main() == 0
    updated = load_config_bundle(
        tmp_path / "configs/experiment.yaml",
        model_path=tmp_path / "configs/models/yolov8.yaml",
        hardware_path=tmp_path / "configs/hardware/rtx4090.yaml",
        run_path=tmp_path / "configs/runs/full.yaml",
    )
    assert updated.batch_size == 16


def test_dry_run_preserves_existing_execution_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_minimal_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    execution_root = tmp_path / "outputs/calibration/executions/existing"
    execution_root.mkdir(parents=True, exist_ok=True)
    (execution_root / "calibration_report.json").write_text(
        json.dumps(
            {
                "common_stable_batch": 16,
                "stable_candidate_count": 3,
                "candidates": [
                    {"model_family": "yolov8", "batch_size": 16, "status": "stable"},
                    {"model_family": "yolov9", "batch_size": 16, "status": "stable"},
                    {"model_family": "yolo26", "batch_size": 16, "status": "stable"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (execution_root / "completed.marker").write_text("done", encoding="utf-8")
    before = (execution_root / "calibration_report.json").read_text(encoding="utf-8")
    monkeypatch.setattr(
        calibrate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/full.yaml",
            output_dir=str(tmp_path / "outputs/calibration"),
            dry_run=True,
            execute=False,
            apply=False,
            resume=False,
            force=False,
            continue_on_error=False,
            print_report=False,
            candidate_batches="8,16",
        ),
    )
    assert calibrate.main() == 0
    assert (execution_root / "calibration_report.json").read_text(encoding="utf-8") == before
