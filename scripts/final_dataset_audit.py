from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.prepare_dataset import (  # noqa: E402
    build_records,
    validate_processed_dataset,
)
from wrist_fracture.data.preparation import classify_label, save_json  # noqa: E402
from wrist_fracture.paths import get_paths  # noqa: E402


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(2, min(8, cpu // 2 or 2))


def _inspect_label_worker(item: tuple[str, Path, Path]) -> tuple[int, int, int, list[str]]:
    split, lbl_dir, img = item
    local_errors: list[str] = []
    lbl = lbl_dir / f"{img.stem}.txt"
    if not lbl.exists():
        local_errors.append(f"{split}:{img.stem}: missing label")
        return 0, 0, 0, local_errors
    text = lbl.read_text(encoding="utf-8").strip()
    if not text:
        return 0, 1, 0, local_errors
    positive = 1
    boxes = 0
    for row in text.splitlines():
        parts = row.split()
        if len(parts) != 5:
            local_errors.append(f"{split}:{img.stem}: invalid label width")
            continue
        try:
            cls, cx, cy, bw, bh = map(float, parts)
        except ValueError:
            local_errors.append(f"{split}:{img.stem}: non-numeric label")
            continue
        if cls != 0:
            local_errors.append(f"{split}:{img.stem}: class id != 0")
        if not all(np.isfinite([cx, cy, bw, bh])):
            local_errors.append(f"{split}:{img.stem}: non-finite coordinate")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            local_errors.append(f"{split}:{img.stem}: center out of bounds")
        if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
            local_errors.append(f"{split}:{img.stem}: size out of bounds")
        boxes += 1
    return positive, 0, boxes, local_errors


def _inspect_image_worker(
    item: tuple[str, Path, dict[str, Path]],
) -> tuple[int, int, int, int, list[str]]:
    split, image_path, source_by_stem = item
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 1, 1, 0, 0, [f"{split}:{image_path.stem}: unreadable"]
    src = source_by_stem.get(image_path.stem)
    if src is not None:
        src_img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if src_img is not None and tuple(src_img.shape) != tuple(img.shape):
            return 0, 0, 1, 0, [f"{split}:{image_path.stem}: dimension mismatch"]
    return 0, 0, 0, 1, []


def _audit_labels(yolo_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    splits: dict[str, dict[str, Any]] = {}
    label_total = 0
    empty_label_count = 0
    for split in ("train", "val", "test"):
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        img_files = sorted(img_dir.glob("*.png"))
        lbl_files = sorted(lbl_dir.glob("*.txt"))
        img_stems = {p.stem for p in img_files}
        lbl_stems = {p.stem for p in lbl_files}
        orphan_images = sorted(img_stems - lbl_stems)
        orphan_labels = sorted(lbl_stems - img_stems)
        if orphan_images:
            errors.append(f"{split}: orphan images")
        if orphan_labels:
            errors.append(f"{split}: orphan labels")
        split_positive = 0
        split_negative = 0
        split_boxes = 0

        with ThreadPoolExecutor(max_workers=_default_workers()) as executor:
            worker_items = ((split, lbl_dir, img) for img in img_files)
            for positive, negative, boxes, local_errors in executor.map(
                _inspect_label_worker, worker_items
            ):
                split_positive += positive
                split_negative += negative
                split_boxes += boxes
                if negative:
                    empty_label_count += negative
                errors.extend(local_errors)
        label_total += len(lbl_files)
        splits[split] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "positive_images": split_positive,
            "negative_images": split_negative,
            "fracture_boxes": split_boxes,
            "orphan_images": len(orphan_images),
            "orphan_labels": len(orphan_labels),
        }
    return {
        "errors": errors,
        "splits": splits,
        "label_files": label_total,
        "positive_images": sum(item["positive_images"] for item in splits.values()),
        "negative_images": sum(item["negative_images"] for item in splits.values()),
        "empty_label_files": empty_label_count,
        "fracture_boxes": sum(item["fracture_boxes"] for item in splits.values()),
    }


def _audit_images(yolo_dir: Path, source_by_stem: dict[str, Path]) -> dict[str, Any]:
    errors: list[str] = []
    corrupt = 0
    wrong_dimensions = 0
    unreadable = 0
    linked_matches = 0
    image_paths = [
        (split, image_path)
        for split in ("train", "val", "test")
        for image_path in (yolo_dir / "images" / split).glob("*.png")
    ]

    with ThreadPoolExecutor(max_workers=_default_workers()) as executor:
        for c, u, w, linked, local_errors in executor.map(
            _inspect_image_worker,
            ((split, image_path, source_by_stem) for split, image_path in image_paths),
        ):
            corrupt += c
            unreadable += u
            wrong_dimensions += w
            linked_matches += linked
            errors.extend(local_errors)
    return {
        "errors": errors,
        "corrupt_images": corrupt,
        "unreadable_images": unreadable,
        "dimension_mismatches": wrong_dimensions,
        "source_link_matches": linked_matches,
    }


def _split_stats(records: list[Any], splits: dict[str, list[Any]]) -> dict[str, Any]:
    total_patients = len({str(r.patient_id) for r in records if r.patient_id is not None})
    total_images = len(records)
    total_boxes = sum(len(r.fracture_boxes) for r in records)
    rows: dict[str, Any] = {}
    for split, items in splits.items():
        patients = {str(r.patient_id) for r in items if r.patient_id is not None}
        images = len(items)
        pos = sum(bool(r.fracture_boxes) for r in items)
        neg = images - pos
        boxes = sum(len(r.fracture_boxes) for r in items)
        rows[split] = {
            "patient_count": len(patients),
            "image_count": images,
            "positive_image_count": pos,
            "negative_image_count": neg,
            "fracture_box_count": boxes,
            "patient_share": round(len(patients) / total_patients, 6),
            "image_share": round(images / total_images, 6),
            "fracture_prevalence": round(pos / images, 6),
            "background_prevalence": round(neg / images, 6),
        }
    rows["totals"] = {
        "patient_count": total_patients,
        "image_count": total_images,
        "fracture_box_count": total_boxes,
    }
    return rows


def main() -> int:
    paths = get_paths()
    raw = paths.raw
    processed = paths.processed
    yolo_dir = processed / "yolo"
    reports = paths.dataset_reports
    workers = _default_workers()
    records, inspection = build_records(
        raw,
        workers=workers,
        hash_workers=workers,
        batch_size=32,
        progress=False,
        force=False,
    )
    split_records = {}
    for split in ("train", "val", "test"):
        stems = {row["stem"] for row in _load_csv(paths.splits / f"{split}_images.csv")}
        split_records[split] = [rec for rec in records if rec.stem in stems]
    source_by_stem = {rec.stem: rec.image_path for rec in records}
    label_audit = _audit_labels(yolo_dir)
    image_audit = _audit_images(yolo_dir, source_by_stem)
    smoke_path = reports / "loader_smoke_report.json"
    smoke = _load_json(smoke_path) if smoke_path.exists() else {"errors": ["smoke report missing"]}
    validation = validate_processed_dataset(
        processed, workers=workers, batch_size=32, progress=False
    )
    lineage = {
        "official_images": 20327,
        "extracted_images": len(records),
        "inspected_images": len(records),
        "converted_images": len(records),
        "processed_images": sum(
            1
            for split in ("train", "val", "test")
            for _ in (yolo_dir / "images" / split).glob("*.png")
        ),
        "label_files": sum(
            1
            for split in ("train", "val", "test")
            for _ in (yolo_dir / "labels" / split).glob("*.txt")
        ),
        "fracture_annotation_count": sum(len(r.all_boxes) for r in records),
        "converted_fracture_boxes": sum(len(r.fracture_boxes) for r in records),
    }
    source_labels = Counter(b.label for rec in records for b in rec.all_boxes)
    report = {
        "lineage": lineage,
        "inspection": inspection.get("stage_times", {}),
        "annotation_policy": {
            "authoritative_format": "pascalvoc",
            "labels_seen": dict(source_labels),
            "fracture_labels": source_labels.get("fracture", 0),
            "excluded_labels": {
                label: count
                for label, count in source_labels.items()
                if classify_label(label) != "fracture"
            },
        },
        "splits": _split_stats(records, split_records),
        "labels": label_audit,
        "images": image_audit,
        "validation": validation,
        "smoke": smoke,
    }
    save_json(reports / "final_dataset_audit.json", report)
    rows: list[dict[str, Any]] = []
    for split, stats in report["splits"].items():
        if split == "totals":
            continue
        rows.append({"split": split, **stats})
    with (reports / "final_dataset_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if label_audit["errors"] or image_audit["errors"] or validation["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
