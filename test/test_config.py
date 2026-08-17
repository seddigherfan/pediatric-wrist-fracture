from pathlib import Path

import pytest

from wrist_fracture.config import (
    ConfigError,
    load_experiment_config,
    load_project_config,
)


def test_load_project_config():
    cfg = load_project_config(Path("configs/project.yaml"))
    assert cfg.name == "pediatric-wrist-fracture"
    assert cfg.seed == 42


def test_invalid_config(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_project_config(p)


def test_load_experiment_config():
    cfg = load_experiment_config(Path("configs/experiment.yaml"))
    assert cfg.model == "yolo26"
    assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == pytest.approx(1.0)
