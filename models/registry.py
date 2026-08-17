from __future__ import annotations

from dataclasses import dataclass

from wrist_fracture.config import MODEL_FAMILIES, ConfigError, ModelConfig


@dataclass(frozen=True)
class ModelSpec:
    family: str
    checkpoint: str
    scale: str
    params_m: float | None = None
    flops_b: float | None = None
    imgsz: int = 640
    nms_free_default: bool = False
    requires_external_repo: bool = False


MODEL_SPECS: dict[str, dict[str, ModelSpec]] = {
    "yolov8": {
        "n": ModelSpec("yolov8", "yolov8n.pt", "n"),
        "s": ModelSpec("yolov8", "yolov8s.pt", "s"),
        "m": ModelSpec("yolov8", "yolov8m.pt", "m"),
        "l": ModelSpec("yolov8", "yolov8l.pt", "l"),
        "x": ModelSpec("yolov8", "yolov8x.pt", "x"),
    },
    "yolov9": {
        "t": ModelSpec(
            "yolov9", "yolov9t.pt", "t", params_m=2.0, flops_b=7.7, requires_external_repo=False
        ),
        "s": ModelSpec(
            "yolov9", "yolov9s.pt", "s", params_m=7.2, flops_b=26.7, requires_external_repo=False
        ),
        "m": ModelSpec(
            "yolov9", "yolov9m.pt", "m", params_m=20.1, flops_b=76.8, requires_external_repo=False
        ),
        "c": ModelSpec(
            "yolov9", "yolov9c.pt", "c", params_m=25.5, flops_b=102.8, requires_external_repo=False
        ),
        "e": ModelSpec(
            "yolov9", "yolov9e.pt", "e", params_m=58.1, flops_b=192.5, requires_external_repo=False
        ),
    },
    "yolo26": {
        "n": ModelSpec("yolo26", "yolo26n.pt", "n", nms_free_default=True),
        "s": ModelSpec("yolo26", "yolo26s.pt", "s", nms_free_default=True),
        "m": ModelSpec("yolo26", "yolo26m.pt", "m", nms_free_default=True),
        "l": ModelSpec("yolo26", "yolo26l.pt", "l", nms_free_default=True),
        "x": ModelSpec("yolo26", "yolo26x.pt", "x", nms_free_default=True),
    },
}


def resolve_model_spec(model: ModelConfig) -> ModelSpec:
    if model.family not in MODEL_FAMILIES:
        raise ConfigError(f"unknown model family: {model.family}")
    family = MODEL_SPECS[model.family]
    if model.scale not in family:
        raise ConfigError(f"unsupported scale {model.scale!r} for family {model.family}")
    spec = family[model.scale]
    if model.checkpoint and model.checkpoint != spec.checkpoint:
        raise ConfigError(
            f"checkpoint mismatch for {model.family}:{model.scale} "
            f"expected {spec.checkpoint}, got {model.checkpoint}"
        )
    return spec


def describe_model_spec(spec: ModelSpec, *, imgsz: int | None = None) -> dict[str, object]:
    return {
        "family": spec.family,
        "checkpoint": spec.checkpoint,
        "scale": spec.scale,
        "params_m": spec.params_m,
        "flops_b": spec.flops_b,
        "imgsz": spec.imgsz if imgsz is None else imgsz,
        "nms_free_default": spec.nms_free_default,
        "requires_external_repo": spec.requires_external_repo,
    }
