from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


MODEL_FAMILIES = {"yolov8", "yolov9", "yolo26"}
SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class ModelConfig:
    family: str
    checkpoint: str
    scale: str
    task: str = "detect"
    implementation: str = "ultralytics"
    supports_training: bool = True
    requires_external_repo: bool = False
    nms_free_default: bool = False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.family == other
        if not isinstance(other, ModelConfig):
            return NotImplemented
        return asdict(self) == asdict(other)


@dataclass(frozen=True)
class HardwareConfig:
    device: str = "cpu"
    amp: bool = False
    workers: int = 0
    cache: str = "ram"
    deterministic: bool = True
    allow_cpu_training: bool = False
    require_gpu: bool = False


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_root: Path
    run_id: str | None = None
    resume: bool = False
    save_period: int = 1
    validation_split: str = "val"
    test_split: str = "test"
    allow_test_evaluation: bool = False
    selection_metric: str = "metrics/mAP50-95(B)"
    repeated_runs: int = 1
    batch_size_policy: str = "fixed"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_yaml: Path
    dataset_split_yaml: Path | None
    model: ModelConfig
    hardware: HardwareConfig
    run: RunConfig
    image_size: int = 640
    epochs: int = 100
    patience: int = 20
    seed: int = 42
    pretrained: bool = True
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    weight_decay: float = 0.0005
    augmentation: dict[str, Any] = field(default_factory=dict)
    resume_checkpoint: Path | None = None
    save_json: bool = True
    batch_size: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def train_ratio(self) -> float:
        return 0.7

    @property
    def val_ratio(self) -> float:
        return 0.15

    @property
    def test_ratio(self) -> float:
        return 0.15


RUN_OVERLAY_KEYS = {
    "image_size",
    "epochs",
    "patience",
    "seed",
    "pretrained",
    "optimizer",
    "lr0",
    "lrf",
    "weight_decay",
    "augmentation",
    "resume_checkpoint",
    "save_json",
    "batch_size",
    "dataset",
    "model",
    "hardware",
    "run",
}


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: Path
    data_dir: Path
    log_level: str = "INFO"


def _require_mapping(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source} must contain a mapping at its root")
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _require_mapping(data, str(config_path))


def _coerce_exp(exp: dict[str, Any]) -> ExperimentConfig:
    model = _require_mapping(exp["model"], "experiment.model")
    hardware = _require_mapping(exp["hardware"], "experiment.hardware")
    run = _require_mapping(exp["run"], "experiment.run")
    dataset = _require_mapping(exp["dataset"], "experiment.dataset")
    return ExperimentConfig(
        dataset_yaml=Path(dataset["yaml"]),
        dataset_split_yaml=Path(dataset["split_yaml"]) if dataset.get("split_yaml") else None,
        model=ModelConfig(
            family=str(model["family"]),
            checkpoint=str(model["checkpoint"]),
            scale=str(model["scale"]),
            task=str(model.get("task", "detect")),
            implementation=str(model.get("implementation", "ultralytics")),
            supports_training=bool(model.get("supports_training", True)),
            requires_external_repo=bool(model.get("requires_external_repo", False)),
            nms_free_default=bool(model.get("nms_free_default", False)),
        ),
        hardware=HardwareConfig(
            device=str(hardware.get("device", "cpu")),
            amp=bool(hardware.get("amp", False)),
            workers=int(hardware.get("workers", 0)),
            cache=str(hardware.get("cache", "ram")),
            deterministic=bool(hardware.get("deterministic", True)),
            allow_cpu_training=bool(hardware.get("allow_cpu_training", False)),
            require_gpu=bool(hardware.get("require_gpu", False)),
        ),
        run=RunConfig(
            name=str(run["name"]),
            output_root=Path(run["output_root"]),
            run_id=str(run.get("run_id")) if run.get("run_id") else None,
            resume=bool(run.get("resume", False)),
            save_period=int(run.get("save_period", 1)),
            validation_split=str(run.get("validation_split", "val")),
            test_split=str(run.get("test_split", "test")),
            allow_test_evaluation=bool(run.get("allow_test_evaluation", False)),
            selection_metric=str(run.get("selection_metric", "metrics/mAP50-95(B)")),
            repeated_runs=int(run.get("repeated_runs", 1)),
            batch_size_policy=str(run.get("batch_size_policy", "fixed")),
        ),
        image_size=int(exp.get("image_size", 640)),
        epochs=int(exp.get("epochs", 100)),
        patience=int(exp.get("patience", 20)),
        seed=int(exp.get("seed", 42)),
        pretrained=bool(exp.get("pretrained", True)),
        optimizer=str(exp.get("optimizer", "auto")),
        lr0=float(exp.get("lr0", 0.01)),
        lrf=float(exp.get("lrf", 0.01)),
        weight_decay=float(exp.get("weight_decay", 0.0005)),
        augmentation=dict(exp.get("augmentation", {})),
        resume_checkpoint=Path(exp["resume_checkpoint"]) if exp.get("resume_checkpoint") else None,
        save_json=bool(exp.get("save_json", True)),
        batch_size=int(exp.get("batch_size", 1)),
        extra={
            k: v
            for k, v in exp.items()
            if k
            not in {
                "dataset",
                "model",
                "hardware",
                "run",
                "image_size",
                "epochs",
                "patience",
                "seed",
                "pretrained",
                "optimizer",
                "lr0",
                "lrf",
                "weight_decay",
                "augmentation",
                "resume_checkpoint",
                "save_json",
                "batch_size",
            }
        },
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = load_yaml_config(path)
    if "experiment" not in data:
        raise ConfigError("experiment section is missing")
    return _coerce_exp(_require_mapping(data["experiment"], "experiment"))


def load_project_config(path: str | Path) -> ProjectConfig:
    data = load_yaml_config(path).get("project")
    if not isinstance(data, dict):
        raise ConfigError("project section is missing or invalid")
    return ProjectConfig(
        name=str(data["name"]),
        seed=int(data["seed"]),
        output_dir=Path(data["output_dir"]),
        data_dir=Path(data["data_dir"]),
        log_level=str(data.get("log_level", "INFO")),
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_bundle(
    experiment_path: str | Path,
    *,
    model_path: str | Path | None = None,
    hardware_path: str | Path | None = None,
    run_path: str | Path | None = None,
) -> ExperimentConfig:
    payload = load_yaml_config(experiment_path)
    experiment = _require_mapping(payload.get("experiment"), "experiment")
    for extra_path in [model_path, hardware_path, run_path]:
        if extra_path:
            extra = load_yaml_config(extra_path)
            if "experiment" in extra:
                experiment = _deep_merge(
                    experiment,
                    _require_mapping(extra["experiment"], str(extra_path)),
                )
            else:
                for section in ("model", "hardware", "run"):
                    if section in extra:
                        experiment[section] = _deep_merge(
                            _require_mapping(experiment.get(section), f"experiment.{section}"),
                            _require_mapping(extra[section], f"{extra_path}.{section}"),
                        )
                for key, value in extra.items():
                    if key in RUN_OVERLAY_KEYS:
                        if isinstance(value, dict) and isinstance(experiment.get(key), dict):
                            experiment[key] = _deep_merge(
                                _require_mapping(experiment.get(key), f"experiment.{key}"),
                                _require_mapping(value, f"{extra_path}.{key}"),
                            )
                        else:
                            experiment[key] = value
    if "experiment" not in payload:
        raise ConfigError("experiment section is missing")
    payload["experiment"] = experiment
    return _coerce_exp(experiment)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    data = asdict(cfg)
    data["dataset_yaml"] = str(cfg.dataset_yaml)
    data["dataset_split_yaml"] = str(cfg.dataset_split_yaml) if cfg.dataset_split_yaml else None
    data["resume_checkpoint"] = str(cfg.resume_checkpoint) if cfg.resume_checkpoint else None
    data["run"]["output_root"] = str(cfg.run.output_root)
    return data


def validate_experiment_config(
    cfg: ExperimentConfig,
    *,
    dry_run: bool = False,
    allow_cpu_smoke: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not cfg.dataset_yaml.exists():
        errors.append(f"missing dataset YAML: {cfg.dataset_yaml}")
    if cfg.dataset_split_yaml is not None and not cfg.dataset_split_yaml.exists():
        errors.append(f"missing split YAML: {cfg.dataset_split_yaml}")
    if cfg.model.family not in MODEL_FAMILIES:
        errors.append(f"unknown model family: {cfg.model.family}")
    if cfg.image_size <= 0:
        errors.append("invalid image size")
    if cfg.epochs <= 0:
        errors.append("invalid epoch count")
    if cfg.batch_size <= 0:
        errors.append("invalid batch size")
    if cfg.hardware.workers < 0:
        errors.append("workers must be non-negative")
    if cfg.hardware.device == "cpu" and cfg.hardware.require_gpu:
        errors.append("incompatible CPU/GPU execution: GPU required")
    if (
        cfg.hardware.device == "cpu"
        and cfg.hardware.allow_cpu_training is False
        and not allow_cpu_smoke
    ):
        errors.append("CPU training is disabled for this run")
    if cfg.run.validation_split not in SPLITS or cfg.run.test_split not in SPLITS:
        errors.append("invalid split name")
    if cfg.run.validation_split == "test":
        errors.append("test split misuse for validation")
    if cfg.run.test_split == cfg.run.validation_split:
        errors.append("validation and test splits must differ")
    if cfg.run.resume and cfg.resume_checkpoint is None:
        errors.append("resume requested without checkpoint")
    if cfg.run.resume and cfg.resume_checkpoint is not None and not cfg.resume_checkpoint.exists():
        errors.append("resume checkpoint missing")
    if cfg.model.family not in MODEL_FAMILIES:
        errors.append("unknown model family")
    if cfg.run.output_root.exists() and not cfg.run.output_root.is_dir():
        errors.append("output collision at output_root")
    if cfg.model.task != "detect":
        errors.append("unsupported task")
    if cfg.model.requires_external_repo:
        errors.append("unknown checkpoint or unsupported external repo model")
    if dry_run and cfg.run.repeated_runs <= 0:
        errors.append("repeated_runs must be positive")
    return errors


def describe_config_composition(
    experiment_path: str | Path,
    *,
    model_path: str | Path | None = None,
    hardware_path: str | Path | None = None,
    run_path: str | Path | None = None,
) -> dict[str, Any]:
    base = load_yaml_config(experiment_path)
    composed = {
        "base": base,
        "resolved": load_config_bundle(
            experiment_path,
            model_path=model_path,
            hardware_path=hardware_path,
            run_path=run_path,
        ),
    }
    return {
        "base": base,
        "resolved": config_to_dict(composed["resolved"]),
    }
