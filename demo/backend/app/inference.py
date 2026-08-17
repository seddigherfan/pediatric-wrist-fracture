from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from .image_utils import render_boxes
from .model_registry import ModelRegistry


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def run_prediction(
    registry: ModelRegistry,
    model_id: str,
    image: Image.Image,
    conf: float,
    iou: float,
    request_dir: Path,
    request_id: str,
) -> dict:
    start = time.perf_counter()
    model = registry.get_model(model_id)
    _sync_cuda()
    infer_start = time.perf_counter()
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        imgsz=registry.config.image_size,
        verbose=False,
        device=registry.config.device
        if registry.config.allow_cpu or "cuda" in registry.config.device
        else "cpu",
    )
    _sync_cuda()
    infer_ms = (time.perf_counter() - infer_start) * 1000
    result = results[0]
    boxes = []
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        boxes.append(
            {
                "class_id": cls,
                "class_name": names.get(cls, "fracture"),
                "class_name_fa": "شکستگی",
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "nx1": x1 / image.width,
                "ny1": y1 / image.height,
                "nx2": x2 / image.width,
                "ny2": y2 / image.height,
            }
        )
    annotated = render_boxes(image, boxes)
    request_dir.mkdir(parents=True, exist_ok=True)
    out_path = request_dir / "annotated.jpg"
    annotated.save(out_path, quality=95)
    total_ms = (time.perf_counter() - start) * 1000
    return {
        "request_id": request_id,
        "inference_time_ms": infer_ms,
        "total_processing_time_ms": total_ms,
        "boxes": boxes,
        "annotated_path": out_path,
    }
