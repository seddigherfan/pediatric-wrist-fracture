from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from wrist_fracture.config import ConfigError

from .config import DemoConfig


@dataclass(frozen=True)
class DemoModelSpec:
    id: str
    display_name: str
    english_name: str
    family: str
    description: str
    default: bool = False


MODEL_SPECS: dict[str, DemoModelSpec] = {
    "yolov8": DemoModelSpec(
        "yolov8", "YOLOv8 — مدل پایه", "YOLOv8 Base", "yolov8", "مدل پایه و مرجع", True
    ),
    "yolov9": DemoModelSpec(
        "yolov9", "YOLOv9 — مدل مقایسه‌ای", "YOLOv9 Comparative", "yolov9", "مدل مقایسه‌ای"
    ),
    "yolo26": DemoModelSpec(
        "yolo26", "YOLO26 — مدل پیشنهادی", "YOLO26 Proposed", "yolo26", "مدل پیشنهادی پژوهش"
    ),
}


def _is_valid_best_pt(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def discover_checkpoint(config: DemoConfig, family: str) -> Path | None:
    explicit = config.model_paths.get(family)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if _is_valid_best_pt(path) else None
    candidates = []
    glob_root = config.project_root / "outputs" / "experiments" / family
    for path in sorted(glob_root.glob("**/checkpoints/best.pt")):
        if "full-" not in path.as_posix():
            continue
        if _is_valid_best_pt(path):
            candidates.append(path)
    if candidates:
        return candidates[-1]
    return None


@dataclass
class RegisteredModel:
    spec: DemoModelSpec
    checkpoint_path: Path | None

    @property
    def available(self) -> bool:
        return self.checkpoint_path is not None


class ModelRegistry:
    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._load_locks: dict[str, threading.Lock] = {
            model_id: threading.Lock() for model_id in MODEL_SPECS
        }
        self._models: dict[str, object] = {}
        self._registered = {
            model_id: RegisteredModel(
                spec=spec, checkpoint_path=discover_checkpoint(config, model_id)
            )
            for model_id, spec in MODEL_SPECS.items()
        }

    def list_models(self) -> list[RegisteredModel]:
        return [self._registered[key] for key in MODEL_SPECS]

    def get_model(self, model_id: str):
        if model_id not in MODEL_SPECS:
            raise ConfigError(f"unknown model id: {model_id}")
        registered = self._registered[model_id]
        if not registered.available:
            raise ConfigError(f"checkpoint unavailable for model: {model_id}")
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
        with self._load_locks[model_id]:
            with self._lock:
                if model_id in self._models:
                    return self._models[model_id]
            from ultralytics import YOLO

            model = YOLO(str(registered.checkpoint_path))
            device = (
                self.config.device
                if self.config.allow_cpu or "cuda" in self.config.device
                else "cpu"
            )
            model.to(device)
            with self._lock:
                self._models.clear()
                self._models[model_id] = model
            return model

    def readiness(self) -> dict[str, bool]:
        return {model_id: registered.available for model_id, registered in self._registered.items()}

    def loaded_ids(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())
