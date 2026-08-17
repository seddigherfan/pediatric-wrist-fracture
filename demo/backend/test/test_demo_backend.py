from __future__ import annotations

import io
from pathlib import Path

from demo.backend.app import main as demo_main
from demo.backend.app.image_utils import render_boxes
from demo.backend.app.model_registry import DemoModelSpec, RegisteredModel
from fastapi.testclient import TestClient
from PIL import Image

client = TestClient(demo_main.app)


def png_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    image = Image.new("RGB", size, "white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_models():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    models = client.get("/api/models")
    assert models.status_code == 200
    assert {item["id"] for item in models.json()} == {"yolov8", "yolov9", "yolo26"}


def test_discover_checkpoint(monkeypatch, tmp_path: Path):
    path = tmp_path / "outputs" / "experiments" / "yolov8" / "full-x" / "checkpoints" / "best.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"pt")
    from demo.backend.app.model_registry import discover_checkpoint

    cfg = demo_main.config
    monkeypatch.setattr(cfg, "project_root", tmp_path)
    monkeypatch.setattr(cfg, "model_paths", {"yolov8": None, "yolov9": None, "yolo26": None})
    assert discover_checkpoint(cfg, "yolov8") == path


def test_invalid_model_rejected():
    response = client.post(
        "/api/predict",
        data={"model_id": "missing", "confidence": "0.25", "iou": "0.7"},
        files={"image": ("demo.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 400


def test_invalid_image_rejected():
    response = client.post(
        "/api/predict",
        data={"model_id": "yolov8", "confidence": "0.25", "iou": "0.7"},
        files={"image": ("demo.txt", b"nope", "text/plain")},
    )
    assert response.status_code == 400


def test_oversized_image_rejected(monkeypatch):
    monkeypatch.setattr(demo_main.config, "max_upload_mb", 0)
    response = client.post(
        "/api/predict",
        data={"model_id": "yolov8", "confidence": "0.25", "iou": "0.7"},
        files={"image": ("demo.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 413


def test_prediction_and_download(monkeypatch, tmp_path: Path):
    class DummyBoxes:
        def __iter__(self):
            class B:
                cls = [0]
                conf = [0.91]

                @property
                def xyxy(self):
                    import torch

                    return torch.tensor([[1.0, 2.0, 30.0, 40.0]])

            yield B()

    class DummyResult:
        names = {0: "fracture"}
        boxes = DummyBoxes()

    class DummyModel:
        def to(self, device):
            return self

        def predict(self, *args, **kwargs):
            return [DummyResult()]

    dummy = RegisteredModel(
        DemoModelSpec("yolov8", "YOLOv8 — مدل پایه", "YOLOv8 Base", "yolov8", "desc", True),
        tmp_path / "best.pt",
    )
    (tmp_path / "best.pt").write_bytes(b"pt")
    monkeypatch.setitem(demo_main.registry._registered, "yolov8", dummy)
    monkeypatch.setattr(demo_main.registry, "get_model", lambda model_id: DummyModel())
    monkeypatch.setattr(
        demo_main.registry, "readiness", lambda: {"yolov8": True, "yolov9": False, "yolo26": False}
    )
    monkeypatch.setattr(demo_main.config, "output_dir", tmp_path)

    response = client.post(
        "/api/predict",
        data={"model_id": "yolov8", "confidence": "0.25", "iou": "0.7"},
        files={"image": ("demo.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["fracture_detected"] is True
    assert payload["num_detections"] == 1
    download = client.get(payload["download_url"])
    assert download.status_code == 200


def test_render_boxes_smoke():
    image = Image.new("RGB", (128, 128), "white")
    result = render_boxes(
        image,
        [
            {"x1": 5, "y1": 8, "x2": 60, "y2": 70, "confidence": 0.9},
        ],
    )
    assert result.size == image.size
