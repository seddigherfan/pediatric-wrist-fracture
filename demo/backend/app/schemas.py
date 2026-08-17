from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    display_name: str
    english_name: str
    family: str
    checkpoint_available: bool
    default: bool = False
    description: str | None = None


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    active_device: str
    loaded_models: list[str]
    model_readiness: dict[str, bool]
    ultralytics_version: str | None
    application_version: str
    warnings: list[str] = Field(default_factory=list)


class BoxResponse(BaseModel):
    class_id: int
    class_name: str
    class_name_fa: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    nx1: float
    ny1: float
    nx2: float
    ny2: float


class PredictResponse(BaseModel):
    request_id: str
    model_id: str
    model_display_name: str
    fracture_detected: bool
    num_detections: int
    maximum_confidence: float
    inference_time_ms: float
    total_processing_time_ms: float
    original_width: int
    original_height: int
    detections: list[BoxResponse]
    annotated_image_url: str
    download_url: str
