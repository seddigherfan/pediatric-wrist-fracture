from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.prepare_dataset import build_records  # noqa: E402


def _write_dataset(root: Path, count: int = 60) -> Path:
    raw = root / "data" / "raw"
    extracted = raw / "extracted" / "set1"
    archives = raw / "archives"
    extracted.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    rows = ["filestem,patient_id,study_id"]
    for idx in range(count):
        stem = f"case_{idx:03d}"
        img = np.full((128, 128), idx % 255, dtype=np.uint8)
        cv2.imwrite(str(extracted / f"{stem}.png"), img)
        (extracted / f"{stem}.xml").write_text(
            f"<annotation><filename>{stem}.png</filename><size><width>128</width><height>128</height></size><object><name>fracture</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>12</ymax></bndbox></object></annotation>",
            encoding="utf-8",
        )
        (extracted / f"{stem}.json").write_text(
            '{"size":{"width":128,"height":128},"objects":[{"classTitle":"fracture","points":{"exterior":[[1,2],[10,12]]}}]}',
            encoding="utf-8",
        )
        rows.append(f"{stem},{idx // 3},{idx}")
    (archives / "dataset.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (archives / "folder_structure.zip").write_text("", encoding="utf-8")
    return raw


def _write_real_subset(root: Path, source_raw: Path, count: int) -> Path:
    raw = root / "data" / "raw"
    extracted = raw / "extracted" / "subset"
    archives = raw / "archives"
    extracted.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    csv_path = next(source_raw.rglob("dataset.csv"), None)
    rows: list[dict[str, str]] = []
    if csv_path and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))[:count]
    else:
        extracted_source = source_raw / "extracted"
        stems = sorted({p.stem for p in extracted_source.rglob("*.png")})[:count]
        rows = [
            {"filestem": stem, "patient_id": str(i // 3), "study_id": str(i)}
            for i, stem in enumerate(stems)
        ]
    out_rows = ["filestem,patient_id,study_id"]
    source_extracted = source_raw / "extracted"
    for row in rows:
        stem = row.get("filestem") or row.get("filename") or row.get("image")
        if not stem:
            continue
        for suffix in (".png", ".xml", ".json"):
            matches = sorted(source_extracted.rglob(f"{stem}{suffix}"))
            if matches:
                shutil.copy2(matches[0], extracted / matches[0].name)
        out_rows.append(f"{stem},{row.get('patient_id', '')},{row.get('study_id', '')}")
    (archives / "dataset.csv").write_text("\n".join(out_rows) + "\n", encoding="utf-8")
    (archives / "folder_structure.zip").write_text("", encoding="utf-8")
    return raw


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real", "actual"], default="actual")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--source-raw", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="inspect-bench-"))
    try:
        if args.mode == "synthetic":
            raw = _write_dataset(root, args.count)
        elif args.mode == "real":
            raw = _write_real_subset(root, args.source_raw, args.count)
        else:
            raw = args.source_raw
        results = []
        for workers in (1, 8):
            t0 = time.perf_counter()
            records, summary = build_records(
                raw, workers=workers, batch_size=8, progress=False, force=True
            )
            elapsed = time.perf_counter() - t0
            results.append(
                {
                    "workers": workers,
                    "items": len(records),
                    "elapsed_s": round(elapsed, 3),
                    "throughput_items_s": round(len(records) / elapsed, 2),
                    "stage_times": summary["stage_times"],
                }
            )
        serial, parallel = results
        serial_t = serial["elapsed_s"]
        parallel_t = parallel["elapsed_s"]
        print(
            json.dumps(
                {
                    "serial": serial,
                    "parallel": parallel,
                    "speedup": round(serial_t / parallel_t, 2),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    run()
