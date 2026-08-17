from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")


def load_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/vazirmatn/Vazirmatn-Regular.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def render_boxes(image: Image.Image, detections: list[dict[str, float | int | str]]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = load_font(18)
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = f"شکستگی {det['confidence']:.0%}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 102, 204), width=4)
        bbox = draw.textbbox((0, 0), label, font=font)
        padding = 6
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        top = max(0, y1 - text_h - padding * 2)
        draw.rounded_rectangle(
            (x1, top, x1 + text_w + padding * 2, top + text_h + padding * 2),
            radius=6,
            fill=(8, 29, 62),
        )
        draw.text((x1 + padding, top + padding), label, fill=(255, 255, 255), font=font)
    return annotated
