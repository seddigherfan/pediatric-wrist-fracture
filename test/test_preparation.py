from __future__ import annotations

import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts import prepare_dataset
from scripts.prepare_dataset import build_records, validate_split

from wrist_fracture.data.preparation import (
    AnnotationBox,
    ImageRecord,
    build_patient_split,
    json_ready,
    parse_pascalvoc,
    parse_supervisely,
    save_json,
)


def _write_synthetic_dataset(root: Path, count: int = 6) -> Path:
    raw = root / "data" / "raw"
    extracted = raw / "extracted" / "set1"
    archives = raw / "archives"
    extracted.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    rows = ["filestem,patient_id,study_id"]
    for idx in range(count):
        stem = f"case_{idx:02d}"
        img = np.full((32, 32), idx + 1, dtype=np.uint8)
        cv2.imwrite(str(extracted / f"{stem}.png"), img)
        xml = extracted / f"{stem}.xml"
        xml.write_text(
            f"<annotation><filename>{stem}.png</filename><size><width>32</width><height>32</height></size><object><name>fracture</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>12</ymax></bndbox></object></annotation>",
            encoding="utf-8",
        )
        js = extracted / f"{stem}.json"
        js.write_text(
            '{"size":{"width":32,"height":32},"objects":[{"classTitle":"fracture","points":{"exterior":[[1,2],[10,12]]}}]}',
            encoding="utf-8",
        )
        rows.append(f"{stem},{idx // 2},{idx}")
    (archives / "dataset.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (archives / "folder_structure.zip").write_text("", encoding="utf-8")
    return raw


def test_parse_pascalvoc(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<annotation><filename>a.png</filename><size><width>100</width><height>200</height></size><object><name>fracture</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>""",
        encoding="utf-8",
    )
    filename, width, height, boxes = parse_pascalvoc(xml)
    assert filename == "a.png"
    assert width == 100 and height == 200
    assert boxes[0].label == "fracture"


def test_parse_supervisely(tmp_path: Path):
    js = tmp_path / "sample.json"
    js.write_text(
        '{"size":{"width":100,"height":200},"objects":[{"classTitle":"fracture","points":{"exterior":[[10,20],[30,40]]}}]}',
        encoding="utf-8",
    )
    width, height, boxes = parse_supervisely(js)
    assert width == 100 and height == 200
    assert boxes[0].xmin == 10 and boxes[0].xmax == 30


def test_yolo_conversion():
    box = AnnotationBox("fracture", 10, 20, 30, 40, "pascalvoc")
    cls, cx, cy, bw, bh = box.to_yolo(100, 200)
    assert cls == 0
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.15)
    assert bw == pytest.approx(0.2)
    assert bh == pytest.approx(0.1)


def test_bbox_validation():
    with pytest.raises(ValueError):
        AnnotationBox("fracture", 30, 20, 10, 40, "pascalvoc").to_yolo(100, 200)


def test_patient_split_and_leakage():
    split = build_patient_split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], seed=123)
    assert sum(len(v) for v in split.values()) == 10
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
    errors = validate_split(
        {
            "train": [{"patient_id": 1}],
            "val": [{"patient_id": 2}],
            "test": [{"patient_id": 1}],
        }
    )
    assert errors


def test_json_ready_serializes_nested_paths(tmp_path: Path):
    report = {
        "manifest": [
            {
                "archive": Path("data/raw/archives/a.zip"),
                "target": tmp_path / "data" / "raw" / "extracted" / "a",
            }
        ],
        "record": ImageRecord(
            stem="sample",
            image_path=tmp_path / "data" / "raw" / "extracted" / "sample.png",
            annotation_path=tmp_path / "data" / "raw" / "extracted" / "sample.xml",
            annotation_format="pascalvoc",
            patient_id="1",
            study_id="2",
            width=100,
            height=200,
            channels=1,
            dtype="uint8",
            fracture_boxes=[AnnotationBox("fracture", 10, 20, 30, 40, "pascalvoc")],
            all_boxes=[AnnotationBox("fracture", 10, 20, 30, 40, "pascalvoc")],
            labels=["fracture"],
        ),
    }

    ready = json_ready(report, base_dir=tmp_path)
    json.dumps(ready)
    assert ready["manifest"][0]["target"] == "data/raw/extracted/a"
    assert ready["record"]["image_path"] == "data/raw/extracted/sample.png"
    assert ready["record"]["annotation_path"] == "data/raw/extracted/sample.xml"


def test_save_json_handles_paths(tmp_path: Path):
    output = tmp_path / "report.json"
    save_json(output, {"path": Path("data") / "nested" / "file.txt"})
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["path"] == "data/nested/file.txt"


def test_inspect_serial_parallel_equivalence(tmp_path: Path):
    raw = _write_synthetic_dataset(tmp_path, count=6)
    serial_records, serial_summary = build_records(raw, workers=1, batch_size=2, progress=False)
    parallel_records, parallel_summary = build_records(raw, workers=2, batch_size=2, progress=False)
    assert [r.stem for r in serial_records] == [r.stem for r in parallel_records]
    assert [r.labels for r in serial_records] == [r.labels for r in parallel_records]
    assert serial_summary["comparison"] == parallel_summary["comparison"]


def test_inspect_checkpoint_reuse_and_invalidation(tmp_path: Path):
    raw = _write_synthetic_dataset(tmp_path, count=4)
    first_records, first_summary = build_records(raw, workers=1, batch_size=2, progress=False)
    second_records, second_summary = build_records(raw, workers=1, batch_size=2, progress=False)
    assert [r.stem for r in first_records] == [r.stem for r in second_records]
    image = raw / "extracted" / "set1" / "case_00.png"
    cv2.imwrite(str(image), np.full((32, 32), 255, dtype=np.uint8))
    third_records, _ = build_records(raw, workers=1, batch_size=2, progress=False)
    assert [r.stem for r in first_records] == [r.stem for r in third_records]


def test_inspect_worker_exception_surfaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw = _write_synthetic_dataset(tmp_path, count=2)

    def boom(path_str: str):
        raise RuntimeError("bad file")

    monkeypatch.setattr("scripts.prepare_dataset._inspect_image_worker", boom)
    with pytest.raises(RuntimeError, match="inspect-images failed"):
        build_records(raw, workers=2, batch_size=1, progress=False, force=True)


def _write_yolo_dataset(root: Path) -> Path:
    yolo = root / "data" / "processed" / "yolo"
    for split in ("train", "val", "test"):
        (yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
    cases = {
        "train": [
            ("train_neg", []),
            (
                "train_multi",
                ["0 0.500000 0.500000 0.250000 0.250000", "0 0.250000 0.250000 0.100000 0.100000"],
            ),
        ],
        "val": [("val_pos", ["0 0.500000 0.500000 0.200000 0.200000"])],
        "test": [("test_neg", [])],
    }
    for split, rows in cases.items():
        for stem, lines in rows:
            img = np.full((48, 64, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(yolo / "images" / split / f"{stem}.png"), img)
            (yolo / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
    (yolo / "dataset.yaml").write_text(
        textwrap.dedent(
            f"""
            path: {yolo.as_posix()}
            train: images/train
            val: images/val
            test: images/test
            names:
              0: fracture
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return yolo


def test_smoke_loader_uses_dataset_yaml_and_splits(tmp_path: Path):
    yolo = _write_yolo_dataset(tmp_path)
    report = prepare_dataset.smoke_load_dataset(yolo, max_batches=2, batch_size=1, workers=0)
    assert report["dataset_yaml"].endswith("dataset.yaml")
    assert set(report["splits"]) == {"train", "val", "test"}
    assert report["splits"]["train"]["loaded_batches"] == 2
    assert report["splits"]["train"]["negative_samples"] >= 1
    assert report["splits"]["train"]["multi_box_samples"] >= 1
    assert report["splits"]["val"]["loaded_batches"] == 1
    assert report["splits"]["test"]["loaded_batches"] == 1
    assert report["splits"]["train"]["images_loaded"] > 0
    assert report["splits"]["train"]["labels_loaded"] >= 0


def test_smoke_loader_bounded_batches(tmp_path: Path):
    yolo = _write_yolo_dataset(tmp_path)
    report = prepare_dataset.smoke_load_dataset(yolo, max_batches=1, batch_size=1, workers=0)
    assert report["splits"]["train"]["loaded_batches"] == 1
    assert len(report["splits"]["train"]["batches"]) == 1


def test_smoke_loader_missing_dataset_yaml_fails(tmp_path: Path):
    yolo = tmp_path / "data" / "processed" / "yolo"
    yolo.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Dataset YAML not found"):
        prepare_dataset.smoke_load_dataset(yolo, max_batches=1, batch_size=1, workers=0)


def test_smoke_loader_invalid_path_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Dataset YAML not found"):
        prepare_dataset.smoke_load_dataset(
            tmp_path / "missing", max_batches=1, batch_size=1, workers=0
        )


def test_smoke_loader_has_no_model_data_dependency():
    source = Path(prepare_dataset.__file__).read_text(encoding="utf-8")
    assert "model.data" not in source
