from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoConfig:
    project_root: Path
    output_dir: Path
    max_upload_mb: int
    confidence_default: float
    iou_default: float
    image_size: int
    device: str
    allow_cpu: bool
    cors_origins: list[str]
    model_paths: dict[str, str | None]
    output_ttl_hours: int


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_config() -> DemoConfig:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(os.getenv("DEMO_OUTPUT_DIR", project_root / "outputs" / "demo")).resolve()
    model_paths = {
        "yolov8": os.getenv("DEMO_MODEL_YOLOV8_PATH"),
        "yolov9": os.getenv("DEMO_MODEL_YOLOV9_PATH"),
        "yolo26": os.getenv("DEMO_MODEL_YOLO26_PATH"),
    }
    return DemoConfig(
        project_root=project_root,
        output_dir=output_dir,
        max_upload_mb=int(os.getenv("DEMO_MAX_UPLOAD_MB", "20")),
        confidence_default=float(os.getenv("DEMO_CONFIDENCE_DEFAULT", "0.25")),
        iou_default=float(os.getenv("DEMO_IOU_DEFAULT", "0.7")),
        image_size=int(os.getenv("DEMO_IMAGE_SIZE", "640")),
        device=os.getenv("DEMO_DEVICE", "cuda:0"),
        allow_cpu=os.getenv("DEMO_ALLOW_CPU", "false").lower() in {"1", "true", "yes", "on"},
        cors_origins=_split_csv(os.getenv("DEMO_CORS_ORIGINS")) or ["http://localhost:3000"],
        model_paths=model_paths,
        output_ttl_hours=int(os.getenv("DEMO_OUTPUT_TTL_HOURS", "24")),
    )
