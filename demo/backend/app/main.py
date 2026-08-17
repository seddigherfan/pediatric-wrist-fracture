from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from wrist_fracture import __version__ as app_version

from .config import load_config
from .image_utils import decode_image
from .inference import run_prediction
from .model_registry import MODEL_SPECS, ModelRegistry
from .schemas import HealthResponse, ModelInfo, PredictResponse

config = load_config()
config.output_dir.mkdir(parents=True, exist_ok=True)
registry = ModelRegistry(config)
app = FastAPI(title="Pediatric Wrist Fracture Demo", version=app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_upload = File(...)
model_form = Form(...)
confidence_form = Form(config.confidence_default)
iou_form = Form(config.iou_default)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device = config.device if cuda_available else "cpu"
    except Exception:
        cuda_available = False
        device = "cpu"
    try:
        import ultralytics

        uv = ultralytics.__version__
    except Exception:
        uv = None
    warnings = []
    if not cuda_available:
        warnings.append("CUDA unavailable; running with CPU fallback only if allowed by config.")
    return HealthResponse(
        status="ok",
        cuda_available=cuda_available,
        active_device=device,
        loaded_models=registry.loaded_ids(),
        model_readiness=registry.readiness(),
        ultralytics_version=uv,
        application_version=app_version,
        warnings=warnings,
    )


@app.get("/api/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    return [
        ModelInfo(
            id=model_id,
            display_name=spec.spec.display_name,
            english_name=spec.spec.english_name,
            family=spec.spec.family,
            checkpoint_available=spec.available,
            default=spec.spec.default,
            description=spec.spec.description,
        )
        for model_id, spec in registry._registered.items()
    ]


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = image_upload,
    model_id: str = model_form,
    confidence: float = confidence_form,
    iou: float = iou_form,
):
    request_id = uuid.uuid4().hex
    if model_id not in MODEL_SPECS:
        raise HTTPException(status_code=400, detail="مدل انتخاب‌شده معتبر نیست.")
    if not registry.readiness()[model_id]:
        raise HTTPException(status_code=409, detail="checkpoint مدل انتخاب‌شده در دسترس نیست.")
    if confidence < 0 or confidence > 1 or iou < 0 or iou > 1:
        raise HTTPException(status_code=400, detail="مقادیر confidence یا IoU نامعتبر است.")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="نوع فایل ارسالی معتبر نیست.")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="فایل تصویر خالی است.")
    if len(raw) > config.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="حجم تصویر بیش از حد مجاز است.")
    ext = Path(image.filename or "").suffix.lower()
    if ext and ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="فرمت تصویر پشتیبانی نمی‌شود.")
    try:
        decoded = decode_image(raw)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="تصویر ارسالی قابل خواندن نیست.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="خطا در پردازش تصویر.") from exc
    request_dir = config.output_dir / request_id
    result = run_prediction(registry, model_id, decoded, confidence, iou, request_dir, request_id)
    rel_url = f"/api/results/{result['request_id']}/annotated"
    return PredictResponse(
        request_id=result["request_id"],
        model_id=model_id,
        model_display_name=registry._registered[model_id].spec.display_name,
        fracture_detected=bool(result["boxes"]),
        num_detections=len(result["boxes"]),
        maximum_confidence=max((box["confidence"] for box in result["boxes"]), default=0.0),
        inference_time_ms=result["inference_time_ms"],
        total_processing_time_ms=result["total_processing_time_ms"],
        original_width=decoded.width,
        original_height=decoded.height,
        detections=result["boxes"],
        annotated_image_url=rel_url,
        download_url=rel_url,
    )


@app.get("/api/results/{request_id}/annotated")
def annotated(request_id: str):
    if "/" in request_id or ".." in request_id:
        raise HTTPException(status_code=400, detail="شناسه نامعتبر است.")
    path = config.output_dir / request_id / "annotated.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="نتیجه یافت نشد.")
    return FileResponse(path, media_type="image/jpeg", filename=f"annotated-{request_id}.jpg")
