from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import pandas as pd

SOURCE_LABELS = {
    "fracture",
    "text",
    "axis",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "metal",
}
FRACTURE_LABELS = {"fracture"}
IGNORED_LABELS = {"text", "axis"}
OTHER_LABELS = {"periostealreaction", "pronatorsign", "softtissue", "metal"}
ALL_LABELS = SOURCE_LABELS


@dataclass(frozen=True)
class AnnotationBox:
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source_format: str
    source_id: str | None = None

    def is_valid(self) -> bool:
        vals = [self.xmin, self.ymin, self.xmax, self.ymax]
        return (
            all(math.isfinite(v) for v in vals) and self.xmax > self.xmin and self.ymax > self.ymin
        )

    def to_yolo(self, width: int, height: int) -> list[float]:
        if width <= 0 or height <= 0:
            raise ValueError("invalid image dimensions")
        if not self.is_valid():
            raise ValueError("invalid box geometry")
        if self.xmin < 0 or self.ymin < 0 or self.xmax > width or self.ymax > height:
            raise ValueError("coordinates outside image")
        cx = ((self.xmin + self.xmax) / 2.0) / width
        cy = ((self.ymin + self.ymax) / 2.0) / height
        bw = (self.xmax - self.xmin) / width
        bh = (self.ymax - self.ymin) / height
        return [0.0, cx, cy, bw, bh]

    def key(self) -> tuple[str, int, int, int, int]:
        return (self.label, round(self.xmin), round(self.ymin), round(self.xmax), round(self.ymax))


@dataclass(frozen=True)
class ImageRecord:
    stem: str
    image_path: Path
    annotation_path: Path | None
    annotation_format: str | None
    patient_id: str | None
    study_id: str | None
    width: int
    height: int
    channels: int
    dtype: str
    fracture_boxes: list[AnnotationBox]
    all_boxes: list[AnnotationBox]
    labels: list[str]
    unreadable: bool = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_to_json(path: Path, base_dir: Path | None = None) -> str:
    candidate = path
    if base_dir is not None:
        try:
            candidate = path.resolve().relative_to(base_dir.resolve())
        except Exception:
            try:
                candidate = path.relative_to(base_dir)
            except Exception:
                candidate = path
        else:
            return candidate.as_posix()
    try:
        root = _project_root()
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def json_ready(value: Any, base_dir: Path | None = None) -> Any:
    if isinstance(value, Path):
        return _path_to_json(value, base_dir=base_dir)
    if dataclass_isinstance(value):
        return json_ready(dataclass_to_dict(value), base_dir=base_dir)
    if isinstance(value, dict):
        return {str(key): json_ready(item, base_dir=base_dir) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item, base_dir=base_dir) for item in value]
    if isinstance(value, set):
        return [json_ready(item, base_dir=base_dir) for item in sorted(value, key=str)]
    if isinstance(value, Counter):
        return {str(key): json_ready(item, base_dir=base_dir) for key, item in value.items()}
    return value


def dataclass_isinstance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return {field: getattr(value, field) for field in value.__dataclass_fields__}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_extract_zip(zip_path: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = (target_dir / member.filename).resolve()
            if target_dir.resolve() not in dest.parents and dest != target_dir.resolve():
                raise ValueError(f"unsafe path in zip: {member.filename}")
            extracted.append(str(dest))
        zf.extractall(target_dir)
    return extracted


def parse_pascalvoc(xml_path: Path) -> tuple[str, int, int, list[AnnotationBox]]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename", default="").strip()
    size = root.find("size")
    width = int(float(size.findtext("width", default="0"))) if size is not None else 0
    height = int(float(size.findtext("height", default="0"))) if size is not None else 0
    boxes: list[AnnotationBox] = []
    for idx, obj in enumerate(root.findall("object")):
        label = (obj.findtext("name", default="") or "").strip()
        b = obj.find("bndbox")
        if b is None:
            continue
        boxes.append(
            AnnotationBox(
                label=label,
                xmin=float(b.findtext("xmin", default="nan")),
                ymin=float(b.findtext("ymin", default="nan")),
                xmax=float(b.findtext("xmax", default="nan")),
                ymax=float(b.findtext("ymax", default="nan")),
                source_format="pascalvoc",
                source_id=f"{xml_path.stem}:{idx}",
            )
        )
    return filename, width, height, boxes


def parse_supervisely(json_path: Path) -> tuple[int, int, list[AnnotationBox]]:
    data = json.loads(read_text(json_path))
    size = data.get("size", {})
    width = int(size.get("width", 0))
    height = int(size.get("height", 0))
    boxes: list[AnnotationBox] = []
    for idx, obj in enumerate(data.get("objects", [])):
        pts = obj.get("points", {}).get("exterior", [])
        if len(pts) != 2:
            continue
        (x1, y1), (x2, y2) = pts
        boxes.append(
            AnnotationBox(
                label=str(obj.get("classTitle", "")).strip(),
                xmin=float(min(x1, x2)),
                ymin=float(min(y1, y2)),
                xmax=float(max(x1, x2)),
                ymax=float(max(y1, y2)),
                source_format="supervisely",
                source_id=f"{json_path.stem}:{idx}",
            )
        )
    return width, height, boxes


def parse_dataset_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def classify_label(label: str) -> str:
    if label in FRACTURE_LABELS:
        return "fracture"
    if label in IGNORED_LABELS:
        return "ignored"
    if label in OTHER_LABELS:
        return "other"
    return "unknown"


def choose_authoritative_format(
    records: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    stats = {
        "pascalvoc_images": 0,
        "supervisely_images": 0,
        "paired_images": 0,
        "matching_boxes": 0,
        "mismatched_boxes": 0,
        "labels_seen": Counter(),
    }
    for rec in records:
        fmt = rec.get("annotation_format")
        if fmt == "pascalvoc":
            stats["pascalvoc_images"] += 1
        elif fmt == "supervisely":
            stats["supervisely_images"] += 1
        if rec.get("paired"):
            stats["paired_images"] += 1
            if rec.get("boxes_match"):
                stats["matching_boxes"] += 1
            else:
                stats["mismatched_boxes"] += 1
        for label in rec.get("labels", []):
            stats["labels_seen"][label] += 1
    authoritative = (
        "pascalvoc" if stats["pascalvoc_images"] >= stats["supervisely_images"] else "supervisely"
    )
    return authoritative, stats


def dataset_structure(root: Path) -> dict[str, Any]:
    image_files = sorted([p for p in root.rglob("*.png") if p.is_file()])
    xml_files = sorted([p for p in root.rglob("*.xml") if p.is_file()])
    json_files = sorted([p for p in root.rglob("*.json") if p.is_file()])
    csv_files = sorted([p for p in root.rglob("*.csv") if p.is_file()])
    return {
        "root": str(root),
        "total_images": len(image_files),
        "annotation_files": len(xml_files) + len(json_files),
        "image_formats": sorted({p.suffix.lower() for p in image_files}),
        "csv_files": [str(p) for p in csv_files],
        "xml_files": len(xml_files),
        "json_files": len(json_files),
    }


def image_info(path: Path) -> dict[str, Any]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"path": str(path), "corrupted": True}
    return {
        "path": str(path),
        "corrupted": False,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "channels": 1 if img.ndim == 2 else int(img.shape[2]),
        "dtype": str(img.dtype),
    }


def build_patient_split(patients: list[Any], seed: int = 42) -> dict[str, list[Any]]:
    rng = random.Random(seed)
    patients = sorted(set(patients), key=lambda x: str(x))
    rng.shuffle(patients)
    n = len(patients)
    n_train = round(n * 0.7)
    n_val = round(n * 0.15)
    n_test = n - n_train - n_val
    return {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val : n_train + n_val + n_test],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def ensure_idempotent_remove(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def dedupe_boxes(boxes: Iterable[AnnotationBox]) -> list[AnnotationBox]:
    seen: set[tuple[str, int, int, int, int]] = set()
    deduped: list[AnnotationBox] = []
    for box in boxes:
        key = box.key()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def render_box_sample(
    image_path: Path, boxes: list[AnnotationBox], out_path: Path, title: str | None = None
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    for box in boxes:
        rect = plt.Rectangle(
            (box.xmin, box.ymin),
            box.xmax - box.xmin,
            box.ymax - box.ymin,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(box.xmin, max(box.ymin - 5, 0), f"{box.label}", color="yellow", fontsize=8)
    if title:
        ax.set_title(title)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def render_histogram(
    values: list[float], out_path: Path, title: str, xlabel: str, ylabel: str = "Count"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=40, color="#355C7D", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
