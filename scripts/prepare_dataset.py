# ruff: noqa: E402
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.download_dataset import download_file, inspect_dataset_record
from wrist_fracture.data.preparation import (
    AnnotationBox,
    ImageRecord,
    build_patient_split,
    classify_label,
    dedupe_boxes,
    parse_dataset_csv,
    parse_pascalvoc,
    parse_supervisely,
    render_histogram,
    safe_extract_zip,
    save_json,
    write_csv,
)
from wrist_fracture.paths import get_paths

PIPELINE_CACHE_SCHEMA = 2


def _now() -> float:
    return time.perf_counter()


def _safe_print(message: str) -> None:
    print(message, flush=True)


def _default_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(8, max(2, cpu // 3)))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_cache_key(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": path.as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _index_dataset_files(extracted: Path) -> dict[str, dict[str, Path]]:
    index = {"images": {}, "xml": {}, "json": {}}
    for path in sorted(extracted.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".png":
            index["images"][path.stem] = path
        elif suffix == ".xml":
            index["xml"][path.stem] = path
        elif suffix == ".json":
            index["json"][path.stem] = path
    return index


def _inspect_image_worker(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    info = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if info is None:
        return {"path": path.as_posix(), "corrupted": True}
    return {
        "path": path.as_posix(),
        "corrupted": False,
        "width": int(info.shape[1]),
        "height": int(info.shape[0]),
        "channels": 1 if info.ndim == 2 else int(info.shape[2]),
        "dtype": str(info.dtype),
    }


def _parse_annotation_worker(item: tuple[str, str]) -> dict[str, Any]:
    kind, path_str = item
    path = Path(path_str)
    if kind == "xml":
        filename, width, height, boxes = parse_pascalvoc(path)
        return {
            "path": path.as_posix(),
            "kind": "xml",
            "filename": filename,
            "width": width,
            "height": height,
            "boxes": [box.key() for box in boxes],
            "raw_boxes": [
                {
                    "label": box.label,
                    "xmin": box.xmin,
                    "ymin": box.ymin,
                    "xmax": box.xmax,
                    "ymax": box.ymax,
                    "source_format": box.source_format,
                    "source_id": box.source_id,
                }
                for box in boxes
            ],
        }
    width, height, boxes = parse_supervisely(path)
    return {
        "path": path.as_posix(),
        "kind": "json",
        "width": width,
        "height": height,
        "boxes": [box.key() for box in boxes],
        "raw_boxes": [
            {
                "label": box.label,
                "xmin": box.xmin,
                "ymin": box.ymin,
                "xmax": box.xmax,
                "ymax": box.ymax,
                "source_format": box.source_format,
                "source_id": box.source_id,
            }
            for box in boxes
        ],
    }


def _sha256_worker(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": path.as_posix(), "sha256": h.hexdigest(), "size": path.stat().st_size}


def _compare_annotation_worker(item: tuple[str, str]) -> dict[str, Any]:
    xml_path, json_path = item
    xml = Path(xml_path)
    js = Path(json_path)
    _, _, _, x = parse_pascalvoc(xml)
    _, _, j = parse_supervisely(js)
    x_d = dedupe_boxes(x)
    j_d = dedupe_boxes(j)
    return {
        "stem": xml.stem,
        "boxes_match": len(x_d) == len(j_d)
        and all(a.key() == b.key() for a, b in zip(x_d, j_d, strict=False)),
        "xml_boxes": len(x_d),
        "json_boxes": len(j_d),
    }


def _load_stage_cache(cache_path: Path) -> dict[str, Any]:
    data = _load_json_file(cache_path)
    if not data or not isinstance(data, dict):
        return {"schema_version": PIPELINE_CACHE_SCHEMA, "entries": {}}
    if data.get("schema_version") != PIPELINE_CACHE_SCHEMA:
        return {"schema_version": PIPELINE_CACHE_SCHEMA, "entries": {}}
    if "entries" not in data or not isinstance(data["entries"], dict):
        return {"schema_version": PIPELINE_CACHE_SCHEMA, "entries": {}}
    return data


def _save_stage_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    payload = {"schema_version": PIPELINE_CACHE_SCHEMA, **payload}
    _atomic_write_text(cache_path, json.dumps(payload, indent=2, sort_keys=True))


def _format_rate(done: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "0.0"
    return f"{done / elapsed:.2f}"


def _stage_status(
    stage: str, done: int, total: int, started: float, workers: int, strategy: str = ""
) -> str:
    elapsed = _now() - started
    eta = ((total - done) / (done / elapsed)) if done and elapsed > 0 else 0.0
    suffix = f" strategy={strategy}" if strategy else ""
    return (
        f"{stage}: {done}/{total} "
        f"rate={_format_rate(done, elapsed)} it/s "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s workers={workers}{suffix}"
    )


def _run_bounded(
    *,
    items: list[Any],
    worker,
    workers: int,
    batch_size: int,
    stage: str,
    progress: bool,
    use_processes: bool,
) -> list[Any]:
    executor_cls = (
        cf.ProcessPoolExecutor if use_processes and workers > 1 else cf.ThreadPoolExecutor
    )
    if workers <= 1:
        iterator = items
        if progress:
            iterator = tqdm(items, desc=stage, unit="item", leave=True)
        return [worker(item) for item in iterator]

    results: list[Any] = []
    with executor_cls(max_workers=workers) as executor:
        iterator = iter(items)
        pending: dict[cf.Future[Any], Any] = {}

        def submit_next() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(worker, item)] = item
            return True

        for _ in range(min(workers * batch_size, len(items))):
            if not submit_next():
                break

        bar = tqdm(total=len(items), desc=stage, unit="item", leave=True) if progress else None
        while pending:
            done, _ = cf.wait(pending.keys(), return_when=cf.FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"{stage} failed for {item}") from exc
                if bar:
                    bar.update(1)
                while len(pending) < workers * batch_size and submit_next():
                    pass
        if bar:
            bar.close()
    return results


def _run_bounded_in_order(
    *,
    items: list[Any],
    worker,
    workers: int,
    batch_size: int,
    stage: str,
    progress: bool,
    use_processes: bool,
) -> list[Any]:
    results = _run_bounded(
        items=items,
        worker=worker,
        workers=workers,
        batch_size=batch_size,
        stage=stage,
        progress=progress,
        use_processes=use_processes,
    )
    return sorted(results, key=lambda item: item.get("order", 0))


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if (
                dst.stat().st_size == src.stat().st_size
                and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns
            ):
                return "existing"
        except FileNotFoundError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _write_text_atomic_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return True


def load_dataset_assets(raw_dir: Path) -> dict[str, Path]:
    archives = raw_dir / "archives"
    extracted = raw_dir / "extracted"
    return {
        "archives": archives,
        "extracted": extracted,
        "dataset_csv": next(archives.rglob("dataset.csv")),
        "folder_structure": next(archives.rglob("folder_structure.zip")),
    }


def extract_archives(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    assets = load_dataset_assets(raw_dir)
    extracted = assets["extracted"]
    extracted.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for zip_path in sorted((raw_dir / "archives").glob("*.zip")):
        target = extracted / zip_path.stem
        if force and target.exists():
            shutil.rmtree(target)
        if target.exists() and any(target.rglob("*")):
            manifest.append({"archive": zip_path.name, "target": str(target), "skipped": True})
            continue
        safe_extract_zip(zip_path, target)
        manifest.append({"archive": zip_path.name, "target": str(target), "skipped": False})
    return {"manifest": manifest, "extracted": extracted}


def locate_annotation_root(extracted: Path) -> tuple[Path | None, Path | None, Path | None]:
    pascal = next(iter(sorted(extracted.rglob("*.xml"))), None)
    sup = next(iter(sorted(extracted.rglob("*.json"))), None)
    images = next(iter(sorted(extracted.rglob("*.png"))), None)
    return (
        pascal.parent if pascal else None,
        sup.parent if sup else None,
        images.parent if images else None,
    )


def build_records(
    raw_dir: Path,
    *,
    workers: int = 1,
    hash_workers: int | None = None,
    batch_size: int = 16,
    progress: bool = False,
    force: bool = False,
) -> tuple[list[ImageRecord], dict[str, Any]]:
    assets = load_dataset_assets(raw_dir)
    csv_df = parse_dataset_csv(assets["dataset_csv"])
    extracted = assets["extracted"]
    inspection_dir = raw_dir.parent / "interim" / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    index = _index_dataset_files(extracted)

    records: list[ImageRecord] = []
    summary = {
        "missing_images": 0,
        "missing_annotations": 0,
        "invalid_images": 0,
        "formats": Counter(),
        "labels": Counter(),
        "stage_times": {},
        "stage_counts": {},
        "duplicate_groups": 0,
        "duplicate_files": 0,
        "hash_mode": "candidate-groups",
    }

    image_cache_path = inspection_dir / "image_inspection_cache.json"
    annotation_cache_path = inspection_dir / "annotation_inspection_cache.json"
    duplicate_cache_path = inspection_dir / "duplicate_index_cache.json"
    comparison_cache_path = inspection_dir / "annotation_comparison_cache.json"

    t0 = _now()
    image_rows: list[dict[str, Any]] = []
    for _, row in csv_df.iterrows():
        stem = str(row.get("filestem") or row.get("filename") or row.get("image"))
        patient_id = str(row.get("patient_id")) if "patient_id" in row else None
        study_id = str(row.get("study_id")) if "study_id" in row else None
        image_path = index["images"].get(stem)
        if image_path is None:
            summary["missing_images"] += 1
            continue
        image_rows.append(
            {
                "stem": stem,
                "patient_id": patient_id,
                "study_id": study_id,
                "image_path": image_path,
                "xml_path": index["xml"].get(stem),
                "json_path": index["json"].get(stem),
            }
        )
    summary["stage_times"]["csv_join"] = _now() - t0
    summary["stage_counts"]["csv_rows"] = len(csv_df)

    image_cache = {} if force else _load_stage_cache(image_cache_path).get("entries", {})
    annotation_cache = {} if force else _load_stage_cache(annotation_cache_path).get("entries", {})

    t1 = _now()
    image_items = []
    image_results: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        path = row["image_path"]
        key = _file_cache_key(path)
        cached = image_cache.get(row["stem"])
        if cached and cached.get("key") == key:
            image_results[row["stem"]] = cached["result"]
        else:
            image_items.append(path.as_posix())
    if image_items:
        image_outputs = _run_bounded(
            items=image_items,
            worker=_inspect_image_worker,
            workers=workers,
            batch_size=batch_size,
            stage="inspect-images",
            progress=progress,
            use_processes=False,
        )
        for output in image_outputs:
            image_results[Path(output["path"]).stem] = output
    summary["stage_times"]["image_inspection"] = _now() - t1
    summary["stage_counts"]["image_items"] = len(image_rows)

    t2 = _now()
    annotation_items: list[tuple[str, str]] = []
    annotation_results: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        ann = row["xml_path"] or row["json_path"]
        if ann is None:
            continue
        kind = "xml" if row["xml_path"] else "json"
        key = _file_cache_key(ann)
        cached = annotation_cache.get(ann.as_posix())
        if cached and cached.get("key") == key:
            annotation_results[row["stem"]] = cached["result"]
        else:
            annotation_items.append((kind, ann.as_posix()))
    if annotation_items:
        annotation_outputs = _run_bounded(
            items=annotation_items,
            worker=_parse_annotation_worker,
            workers=workers,
            batch_size=batch_size,
            stage="parse-annotations",
            progress=progress,
            use_processes=True,
        )
        for output in annotation_outputs:
            annotation_results[Path(output["path"]).stem] = output
    summary["stage_times"]["annotation_parsing"] = _now() - t2
    summary["stage_counts"]["annotation_items"] = len(annotation_items) + len(annotation_results)

    t3 = _now()
    comparable_pairs = [
        (row["xml_path"].as_posix(), row["json_path"].as_posix(), row["stem"])
        for row in image_rows
        if row["xml_path"] and row["json_path"]
    ]
    comparison_results = []
    comparison_cache = {} if force else _load_stage_cache(comparison_cache_path).get("entries", {})
    compare_items = []
    for xml_path, json_path, _stem in comparable_pairs:
        key = f"{xml_path}|{json_path}"
        cached = comparison_cache.get(key)
        if (
            cached
            and cached.get("xml_key") == _file_cache_key(Path(xml_path))
            and cached.get("json_key") == _file_cache_key(Path(json_path))
        ):
            comparison_results.append(cached["result"])
        else:
            compare_items.append((xml_path, json_path))
    if compare_items:
        comparison_results.extend(
            _run_bounded(
                items=compare_items,
                worker=_compare_annotation_worker,
                workers=workers,
                batch_size=batch_size,
                stage="compare-annotations",
                progress=progress,
                use_processes=True,
            )
        )
    summary["stage_times"]["annotation_comparison"] = _now() - t3
    summary["stage_counts"]["comparison_pairs"] = len(comparable_pairs)

    t4 = _now()
    all_files = [row["image_path"] for row in image_rows]
    size_groups: dict[int, list[Path]] = defaultdict(list)
    for path in all_files:
        size_groups[path.stat().st_size].append(path)
    candidate_groups = [paths for paths in size_groups.values() if len(paths) > 1]
    hash_items = [p.as_posix() for group in candidate_groups for p in group]
    hash_results: list[dict[str, Any]] = []
    if hash_items:
        hash_results = _run_bounded(
            items=hash_items,
            worker=_sha256_worker,
            workers=hash_workers or workers,
            batch_size=batch_size,
            stage="hash-candidates",
            progress=progress,
            use_processes=True,
        )
    hash_by_path = {r["path"]: r for r in hash_results}
    duplicates: list[dict[str, Any]] = []
    for group in candidate_groups:
        groups_by_hash: dict[str, list[str]] = defaultdict(list)
        for path in group:
            result = hash_by_path.get(path.as_posix())
            if result:
                groups_by_hash[result["sha256"]].append(path.as_posix())
        for paths in groups_by_hash.values():
            if len(paths) > 1:
                duplicates.append({"paths": sorted(paths), "size": Path(paths[0]).stat().st_size})
    summary["duplicate_groups"] = len(duplicates)
    summary["duplicate_files"] = sum(len(group["paths"]) for group in duplicates)
    summary["stage_times"]["duplicate_screening"] = _now() - t4

    for row in image_rows:
        image_info = image_results.get(row["stem"], {})
        ann_result = annotation_results.get(row["stem"])
        boxes: list[AnnotationBox] = []
        fmt = None
        ann_path = row["xml_path"] or row["json_path"]
        if ann_result:
            fmt = ann_result["kind"]
            boxes = [AnnotationBox(**box) for box in ann_result.get("raw_boxes", [])]
        if not boxes:
            summary["missing_annotations"] += 1
        boxes = dedupe_boxes(boxes)
        fracture_boxes = [b for b in boxes if classify_label(b.label) == "fracture"]
        summary["formats"][fmt or "missing"] += 1
        for b in boxes:
            summary["labels"][b.label] += 1
        if image_info.get("corrupted"):
            summary["invalid_images"] += 1
        records.append(
            ImageRecord(
                row["stem"],
                row["image_path"],
                ann_path,
                "pascalvoc" if fmt == "xml" else "supervisely" if fmt == "json" else None,
                row["patient_id"],
                row["study_id"],
                int(image_info.get("width", 0)),
                int(image_info.get("height", 0)),
                int(image_info.get("channels", 0)),
                str(image_info.get("dtype", "")),
                fracture_boxes,
                boxes,
                sorted({b.label for b in boxes}),
                unreadable=bool(image_info.get("corrupted")),
            )
        )

    image_cache_entries = {
        row["stem"]: {
            "key": _file_cache_key(row["image_path"]),
            "result": image_results[row["stem"]],
        }
        for row in image_rows
        if row["stem"] in image_results
    }
    annotation_cache_entries = {
        (row["xml_path"].as_posix() if row["xml_path"] else row["json_path"].as_posix()): {
            "key": _file_cache_key(row["xml_path"] or row["json_path"]),
            "result": annotation_results[row["stem"]],
        }
        for row in image_rows
        if row["stem"] in annotation_results
    }
    comparison_cache_entries: dict[str, Any] = {}
    for result in comparison_results:
        stem = result["stem"]
        pair = next((pair for pair in comparable_pairs if pair[2] == stem), None)
        if pair is None:
            continue
        xml_path, json_path, _ = pair
        comparison_cache_entries[f"{xml_path}|{json_path}"] = {
            "xml_key": _file_cache_key(Path(xml_path)),
            "json_key": _file_cache_key(Path(json_path)),
            "result": result,
        }
    duplicate_cache_entries = {
        item["path"]: {
            "key": _file_cache_key(Path(item["path"])),
            "result": item,
        }
        for item in hash_results
    }
    _save_stage_cache(image_cache_path, {"entries": image_cache_entries})
    _save_stage_cache(annotation_cache_path, {"entries": annotation_cache_entries})
    _save_stage_cache(comparison_cache_path, {"entries": comparison_cache_entries})
    _save_stage_cache(duplicate_cache_path, {"entries": duplicate_cache_entries})
    summary["comparison"] = {
        "paired_images": len(comparable_pairs),
        "sample_compared": len(comparison_results),
        "matching": sum(1 for item in comparison_results if item["boxes_match"]),
        "mismatching": sum(1 for item in comparison_results if not item["boxes_match"]),
        "xml_only": len(index["xml"]) - len(comparable_pairs),
        "json_only": len(index["json"]) - len(comparable_pairs),
    }
    summary["stage_times"]["total"] = _now() - t0
    return records, summary


def compare_annotation_formats(raw_dir: Path) -> dict[str, Any]:
    extracted = load_dataset_assets(raw_dir)["extracted"]
    xmls = {p.stem: p for p in extracted.rglob("*.xml")}
    jsons = {p.stem: p for p in extracted.rglob("*.json")}
    paired = sorted(set(xmls) & set(jsons))
    sample = paired[:500]
    match = 0
    mismatch = 0
    for stem in sample:
        _, _, _, x = parse_pascalvoc(xmls[stem])
        _, _, j = parse_supervisely(jsons[stem])
        if len(x) == len(j) and all(
            a.key() == b.key() for a, b in zip(dedupe_boxes(x), dedupe_boxes(j), strict=False)
        ):
            match += 1
        else:
            mismatch += 1
    return {
        "paired_images": len(paired),
        "sample_compared": len(sample),
        "matching": match,
        "mismatching": mismatch,
        "xml_only": len(xmls) - len(paired),
        "json_only": len(jsons) - len(paired),
    }


def _convert_record_worker(item: tuple[int, ImageRecord, bool]) -> dict[str, Any]:
    order, rec, negative_empty = item
    lines: list[str] = []
    invalid = 0
    for box in rec.fracture_boxes:
        try:
            cls, cx, cy, bw, bh = box.to_yolo(rec.width, rec.height)
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        except ValueError:
            invalid += 1
    return {
        "order": order,
        "stem": rec.stem,
        "label_text": "\n".join(lines) + ("\n" if lines else ""),
        "has_label": bool(lines),
        "invalid_boxes": invalid,
        "patient_id": rec.patient_id,
        "study_id": rec.study_id,
        "source_annotation": str(rec.annotation_path) if rec.annotation_path else None,
        "source_format": rec.annotation_format,
        "image_path": rec.image_path.as_posix(),
        "fracture_boxes": len(lines),
        "negative_empty": negative_empty,
    }


def _materialize_task_worker(
    item: tuple[str, str, str, str, str, list[str], bool],
) -> dict[str, Any]:
    split, src_path, dst_img, dst_lbl, stem, lines, negative_empty = item
    method = _link_or_copy(Path(src_path), Path(dst_img))
    label_text = "\n".join(lines) + ("\n" if lines else "")
    if lines or negative_empty:
        _write_text_atomic_if_changed(Path(dst_lbl), label_text)
    return {"split": split, "stem": stem, "copy_method": method, "lines": len(lines)}


def convert_to_yolo(
    records: list[ImageRecord],
    processed_dir: Path,
    *,
    workers: int = 1,
    batch_size: int = 16,
    progress: bool = False,
    force: bool = False,
    negative_empty: bool = True,
) -> dict[str, Any]:
    cache_dir = processed_dir.parent / "interim" / "conversion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_cache = cache_dir / "conversion_manifest.json"
    images_root = processed_dir / "images"
    labels_root = processed_dir / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    cached_manifest = {} if force else _load_stage_cache(manifest_cache).get("entries", {})
    tasks: list[tuple[int, ImageRecord, bool]] = []
    for order, rec in enumerate(records):
        label_path = labels_root / f"{rec.stem}.txt"
        img_dst = images_root / f"{rec.stem}.png"
        cache_key = {
            "image": _file_cache_key(rec.image_path),
            "label_count": len(rec.fracture_boxes),
            "width": rec.width,
            "height": rec.height,
            "negative_empty": negative_empty,
        }
        cached = cached_manifest.get(rec.stem)
        if (
            not force
            and cached
            and cached.get("key") == cache_key
            and img_dst.exists()
            and label_path.exists()
        ):
            continue
        tasks.append((order, rec, negative_empty))

    started = _now()
    strategy = "hardlink->copy"
    rows = _run_bounded_in_order(
        items=tasks,
        worker=_convert_record_worker,
        workers=workers,
        batch_size=batch_size,
        stage="convert",
        progress=progress,
        use_processes=False,
    )
    counts = Counter()
    manifest: list[dict[str, Any]] = []
    manifest_entries: dict[str, Any] = {}
    record_by_stem = {rec.stem: rec for rec in records}
    for item in rows:
        rec = record_by_stem[item["stem"]]
        img_dst = images_root / f"{rec.stem}.png"
        lbl_dst = labels_root / f"{rec.stem}.txt"
        method = _link_or_copy(rec.image_path, img_dst)
        strategy = method if strategy == "hardlink->copy" else strategy
        if item["has_label"] or negative_empty:
            _write_text_atomic_if_changed(lbl_dst, item["label_text"])
        counts["images"] += 1
        counts["positive_images" if item["has_label"] else "negative_images"] += 1
        counts["fracture_boxes"] += item["fracture_boxes"]
        manifest_row = {
            "stem": rec.stem,
            "image_path": str(img_dst),
            "label_path": str(lbl_dst),
            "patient_id": rec.patient_id,
            "study_id": rec.study_id,
            "fracture_boxes": item["fracture_boxes"],
            "invalid_boxes": item["invalid_boxes"],
            "source_annotation": item["source_annotation"],
            "source_format": item["source_format"],
            "copy_method": method,
        }
        manifest.append(manifest_row)
        cache_key = {
            "image": _file_cache_key(rec.image_path),
            "label_count": len(rec.fracture_boxes),
            "width": rec.width,
            "height": rec.height,
            "negative_empty": negative_empty,
        }
        manifest_entries[rec.stem] = {
            "key": cache_key,
            "result": manifest_row,
        }

    if not manifest and not force:
        existing = _load_stage_cache(manifest_cache).get("entries", {})
        manifest = [item["result"] for item in existing.values()]
        manifest.sort(key=lambda row: row["stem"])
    else:
        _save_stage_cache(manifest_cache, {"entries": manifest_entries})
        manifest.sort(key=lambda row: row["stem"])
        write_csv(processed_dir / "manifests" / "conversion_manifest.csv", manifest)
    return {
        "counts": dict(counts),
        "manifest_rows": len(manifest),
        "images_dir": str(images_root),
        "labels_dir": str(labels_root),
        "copy_strategy": strategy,
        "elapsed_s": _now() - started,
        "workers": workers,
        "batch_size": batch_size,
    }


def make_splits(records: list[ImageRecord], seed: int = 42) -> dict[str, list[ImageRecord]]:
    by_patient: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        by_patient[str(rec.patient_id)].append(rec)
    split_patients = build_patient_split(list(by_patient.keys()), seed=seed)
    return {k: [rec for pid in v for rec in by_patient[pid]] for k, v in split_patients.items()}


def save_split_files(splits: dict[str, list[ImageRecord]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, rows in splits.items():
        patients = sorted({r.patient_id for r in rows if r.patient_id is not None})
        write_csv(out_dir / f"{split}_patients.csv", [{"patient_id": p} for p in patients])
        write_csv(
            out_dir / f"{split}_images.csv",
            [
                {
                    "stem": r.stem,
                    "patient_id": r.patient_id,
                    "study_id": r.study_id,
                    "image_path": str(r.image_path),
                }
                for r in rows
            ],
        )
        result[split] = {
            "patients": len(patients),
            "images": len(rows),
            "positive_images": sum(bool(r.fracture_boxes) for r in rows),
            "negative_images": sum(not r.fracture_boxes for r in rows),
            "fracture_boxes": sum(len(r.fracture_boxes) for r in rows),
        }
    return result


def validate_records(
    records: list[ImageRecord], split_records: dict[str, list[ImageRecord]] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    box_errors = 0
    for rec in records:
        for box in rec.fracture_boxes:
            try:
                box.to_yolo(rec.width, rec.height)
            except Exception:
                box_errors += 1
    if split_records:
        patient_sets = {k: {r.patient_id for r in v} for k, v in split_records.items()}
        if (
            patient_sets["train"] & patient_sets["val"]
            or patient_sets["train"] & patient_sets["test"]
            or patient_sets["val"] & patient_sets["test"]
        ):
            errors.append("patient leakage detected")
    return {"errors": errors, "invalid_boxes": box_errors, "image_count": len(records)}


def validate_processed_dataset(
    processed_dir: Path,
    workers: int = 1,
    batch_size: int = 64,
    progress: bool = False,
) -> dict[str, Any]:
    yolo_dir = processed_dir / "yolo"
    manifest = processed_dir / "manifests" / "conversion_manifest.csv"
    issues: list[str] = []
    if not yolo_dir.exists():
        return {"errors": ["processed dataset missing"], "valid": False}
    manifest_rows = []
    if manifest.exists():
        import csv

        with manifest.open("r", encoding="utf-8", newline="") as fh:
            manifest_rows = list(csv.DictReader(fh))
    splits = {}
    for split in ("train", "val", "test"):
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        splits[split] = {
            "images": len(list(img_dir.glob("*.png"))),
            "labels": len(list(lbl_dir.glob("*.txt"))),
        }
        if splits[split]["images"] != splits[split]["labels"]:
            issues.append(f"{split} image/label count mismatch")
    if not (yolo_dir / "dataset.yaml").exists():
        issues.append("dataset.yaml missing")
    return {
        "errors": issues,
        "valid": not issues,
        "manifest_rows": len(manifest_rows),
        "splits": splits,
        "workers": workers,
        "batch_size": batch_size,
        "progress": progress,
    }


def generate_dataset_figures(
    records: list[ImageRecord],
    figure_dir: Path,
    seed: int = 42,
    *,
    force: bool = False,
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache_path = figure_dir / "figure_manifest.json"
    existing = {} if force else _load_stage_cache(cache_path).get("entries", {})
    source_labels = Counter(b.label for rec in records for b in rec.all_boxes)
    pos = [1 if rec.fracture_boxes else 0 for rec in records]
    figures = {
        "source_label_distribution.png": (
            list(source_labels.values()),
            "Source Label Distribution",
            "Count per label",
        ),
        "image_width_distribution.png": (
            [rec.width for rec in records],
            "Image Width Distribution",
            "Width (px)",
        ),
        "image_height_distribution.png": (
            [rec.height for rec in records],
            "Image Height Distribution",
            "Height (px)",
        ),
        "aspect_ratio_distribution.png": (
            [rec.width / rec.height for rec in records if rec.height],
            "Aspect Ratio Distribution",
            "Width / Height",
        ),
        "fracture_boxes_per_image.png": (
            [len(rec.fracture_boxes) for rec in records],
            "Fracture Boxes per Image",
            "Boxes",
        ),
        "fracture_bbox_area_distribution.png": (
            [
                sum((b.xmax - b.xmin) * (b.ymax - b.ymin) for b in rec.fracture_boxes)
                for rec in records
            ],
            "Fracture Bounding Box Area Distribution",
            "Pixel area",
        ),
        "positive_negative_distribution.png": (pos, "Positive vs Negative Images", "Binary label"),
    }
    for name, (values, title, xlabel) in figures.items():
        out = figure_dir / name
        if not force and out.exists() and existing.get(name):
            continue
        render_histogram(values, out, title, xlabel)
    _save_stage_cache(
        cache_path,
        {"entries": {name: {"count": len(values)} for name, (values, _, _) in figures.items()}},
    )
    return [str(p) for p in sorted(figure_dir.glob("*.png"))]


def write_dataset_yaml(processed_dir: Path) -> Path:
    yaml_path = processed_dir / "dataset.yaml"
    payload = {
        "path": str(processed_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["fracture"],
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def _load_ultralytics_detection_api() -> tuple[Any, Any, Any]:
    try:
        from ultralytics.data.build import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
    except Exception as exc:  # pragma: no cover - dependency/version failure
        raise RuntimeError(
            "Ultralytics detection dataset APIs are unavailable. "
            "Expected ultralytics.data.utils.check_det_dataset and "
            "ultralytics.data.build.build_yolo_dataset/build_dataloader."
        ) from exc
    return check_det_dataset, build_yolo_dataset, build_dataloader


def _resolve_dataset_yaml(yolo_dir: Path) -> Path:
    yaml_path = yolo_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    return yaml_path


def _read_dataset_names(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError("dataset.yaml must define 'names' as a list or mapping")


def _batch_boxes(batch: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    boxes = batch.get("bboxes")
    cls = batch.get("cls")
    if boxes is None or cls is None:
        return None
    boxes_arr = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    cls_arr = (
        cls.detach().cpu().numpy().reshape(-1)
        if hasattr(cls, "detach")
        else np.asarray(cls).reshape(-1)
    )
    return boxes_arr, cls_arr


def smoke_load_dataset(
    yolo_dir: Path,
    max_batches: int = 2,
    batch_size: int = 2,
    workers: int | None = None,
) -> dict[str, Any]:
    from ultralytics.cfg import get_cfg

    check_det_dataset, build_yolo_dataset, build_dataloader = _load_ultralytics_detection_api()
    yaml_path = _resolve_dataset_yaml(yolo_dir)
    start = _now()
    data = check_det_dataset(str(yaml_path), autodownload=False)
    dataset_yaml = Path(data.get("yaml_file", yaml_path))
    names = _read_dataset_names(data)
    version = __import__("ultralytics").__version__
    worker_count = _default_workers() if workers is None else max(0, int(workers))
    split_stats: dict[str, Any] = {
        "ultralytics_version": version,
        "dataset_yaml": str(dataset_yaml),
        "resolved_dataset_yaml": str(dataset_yaml.resolve()),
        "max_batches": int(max_batches),
        "batch_size": int(batch_size),
        "workers": worker_count,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_path = data.get(split)
        if not split_path:
            raise FileNotFoundError(f"{yaml_path} is missing a usable '{split}' split")
        split_start = _now()
        dataset = build_yolo_dataset(
            cfg=get_cfg(
                overrides={
                    "task": "detect",
                    "imgsz": 64,
                    "rect": False,
                    "cache": False,
                    "single_cls": False,
                    "fraction": 1.0,
                    "classes": None,
                }
            ),
            img_path=str(split_path),
            batch=max(1, int(batch_size)),
            data=data,
            mode=split,
            rect=False,
            stride=32,
            multi_modal=False,
            fraction=1.0,
        )
        labels = dataset.get_labels()
        label_files = getattr(dataset, "label_files", [])
        cache_path = Path(label_files[0]).parent.with_suffix(".cache") if label_files else None
        cache_created_at = (
            cache_path.stat().st_mtime_ns if cache_path and cache_path.exists() else None
        )
        loader = build_dataloader(
            dataset=dataset,
            batch=max(1, int(batch_size)),
            workers=worker_count,
            shuffle=False,
            rank=-1,
            drop_last=False,
            pin_memory=False,
        )
        split_info: dict[str, Any] = {
            "image_path": str(split_path),
            "image_count": len(labels),
            "label_count": len(label_files),
            "initialization_seconds": round(_now() - split_start, 6),
            "cache_created_at_ns": cache_created_at,
            "batches": [],
            "negative_samples": 0,
            "multi_box_samples": 0,
        }
        loader_iter = iter(loader)
        for batch_idx in range(max_batches):
            batch_start = _now()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            boxes = _batch_boxes(batch)
            images = batch.get("img")
            if images is None:
                raise RuntimeError(f"Ultralytics batch for split '{split}' did not include 'img'")
            image_shape = tuple(int(x) for x in getattr(images, "shape", ()))
            if len(image_shape) < 3:
                raise RuntimeError(
                    f"Unexpected image tensor shape for split '{split}': {image_shape}"
                )
            if hasattr(images, "dtype"):
                _ = images.dtype
            if boxes is not None:
                boxes_arr, cls_arr = boxes
                if boxes_arr.size:
                    if not np.isfinite(boxes_arr).all():
                        raise ValueError(
                            f"Non-finite boxes found in split '{split}' batch {batch_idx}"
                        )
                    if np.any((boxes_arr < 0) | (boxes_arr > 1)):
                        raise ValueError(
                            f"Normalized boxes out of range in split '{split}' batch {batch_idx}"
                        )
                    if np.any(cls_arr < 0) or np.any(cls_arr >= len(names)):
                        raise ValueError(
                            f"Negative class ids found in split '{split}' batch {batch_idx}"
                        )
            if "im_file" in batch:
                batch_files = batch["im_file"]
                if isinstance(batch_files, (list, tuple)):
                    batch_sample_count = len(batch_files)
                else:
                    batch_sample_count = (
                        int(getattr(batch_files, "shape", [0])[0])
                        if getattr(batch_files, "shape", None)
                        else 0
                    )
            else:
                batch_sample_count = int(image_shape[0])
            label_slice = dataset.labels[
                batch_idx * max(1, batch_size) : batch_idx * max(1, batch_size) + batch_sample_count
            ]
            for item in label_slice:
                nboxes = len(item.get("cls", []))
                if nboxes == 0:
                    split_info["negative_samples"] += 1
                elif nboxes > 1:
                    split_info["multi_box_samples"] += 1
            elapsed = round(_now() - batch_start, 6)
            if "first_batch_seconds" not in split_info:
                split_info["first_batch_seconds"] = elapsed
            else:
                split_info.setdefault("subsequent_batch_seconds", elapsed)
            split_info["batches"].append(
                {
                    "batch_index": batch_idx,
                    "images": int(image_shape[0]),
                    "shape": list(image_shape),
                    "labels": int(len(batch.get("cls", []))) if batch.get("cls") is not None else 0,
                    "boxes": int(boxes[0].shape[0]) if boxes is not None else 0,
                }
            )
            if batch_idx + 1 >= max_batches:
                break
        if "first_batch_seconds" not in split_info:
            split_info["first_batch_seconds"] = None
        if "subsequent_batch_seconds" not in split_info:
            split_info["subsequent_batch_seconds"] = None
        split_info["loaded_batches"] = len(split_info["batches"])
        split_info["images_loaded"] = int(sum(b["images"] for b in split_info["batches"]))
        split_info["labels_loaded"] = int(sum(b["labels"] for b in split_info["batches"]))
        split_info["load_seconds"] = round(_now() - split_start, 6)
        split_stats["splits"][split] = split_info
    split_stats["total_seconds"] = round(_now() - start, 6)
    return split_stats


def build_final_dataset(
    records: list[ImageRecord],
    processed_dir: Path,
    splits: dict[str, list[ImageRecord]],
    force: bool = False,
    workers: int = 1,
    batch_size: int = 16,
    progress: bool = False,
) -> Path:
    yolo_dir = processed_dir / "yolo"
    if force and yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val", "test"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    tasks = []
    for split, rows in splits.items():
        for rec in rows:
            lines: list[str] = []
            for box in rec.fracture_boxes:
                try:
                    cls, cx, cy, bw, bh = box.to_yolo(rec.width, rec.height)
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                except ValueError:
                    pass
            tasks.append(
                (
                    split,
                    rec.image_path.as_posix(),
                    (yolo_dir / "images" / split / rec.image_path.name).as_posix(),
                    (yolo_dir / "labels" / split / f"{rec.stem}.txt").as_posix(),
                    rec.stem,
                    lines,
                    True,
                )
            )
    _run_bounded(
        items=tasks,
        worker=_materialize_task_worker,
        workers=workers,
        batch_size=batch_size,
        stage="materialize",
        progress=progress,
        use_processes=False,
    )
    write_dataset_yaml(yolo_dir)
    return yolo_dir


def summarize(records: list[ImageRecord], comparison: dict[str, Any]) -> dict[str, Any]:
    all_labels = Counter(b.label for rec in records for b in rec.all_boxes)
    frac = Counter(len(rec.fracture_boxes) for rec in records)
    return {
        "total_images": len(records),
        "patient_count": len({r.patient_id for r in records if r.patient_id is not None}),
        "study_count": len({r.study_id for r in records if r.study_id is not None}),
        "labels": dict(all_labels),
        "fracture_images": sum(bool(r.fracture_boxes) for r in records),
        "negative_images": sum(not r.fracture_boxes for r in records),
        "multiple_fractures": sum(len(r.fracture_boxes) > 1 for r in records),
        "annotation_comparison": comparison,
        "fracture_boxes_per_image": dict(frac),
    }


def cmd_download(args: argparse.Namespace) -> int:
    record = inspect_dataset_record()
    files = record["files"]
    manifest = [download_file(file, args.output_dir, force=args.force) for file in files]
    save_json(args.manifest, {"article": record["article"], "files": manifest})
    return 0


def validate_split(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    patient_sets = {name: {r["patient_id"] for r in rows} for name, rows in splits.items()}
    errors = []
    if patient_sets["train"] & patient_sets["val"]:
        errors.append("train/val overlap")
    if patient_sets["train"] & patient_sets["test"]:
        errors.append("train/test overlap")
    if patient_sets["val"] & patient_sets["test"]:
        errors.append("val/test overlap")
    return errors


def cmd_extract(args: argparse.Namespace) -> int:
    result = extract_archives(args.raw_dir, force=args.force)
    save_json(args.manifest, result)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    started = _now()
    _safe_print(
        "inspect: "
        f"raw_dir={args.raw_dir} "
        f"workers={args.workers} "
        f"hash_workers={args.hash_workers or args.workers} "
        f"batch_size={args.batch_size}"
    )
    records, summary = build_records(
        args.raw_dir,
        workers=args.workers,
        hash_workers=args.hash_workers,
        batch_size=args.batch_size,
        progress=args.progress,
        force=args.force,
    )
    comparison = summary.get("comparison", compare_annotation_formats(args.raw_dir))
    report = summarize(records, comparison)
    report["profiling"] = {
        "stage_times": summary.get("stage_times", {}),
        "stage_counts": summary.get("stage_counts", {}),
        "duplicate_groups": summary.get("duplicate_groups", 0),
        "duplicate_files": summary.get("duplicate_files", 0),
        "hash_mode": summary.get("hash_mode"),
        "workers": args.workers,
        "hash_workers": args.hash_workers or args.workers,
    }
    save_json(args.report_json, report)
    write_csv(
        args.report_csv,
        [
            {
                "stem": r.stem,
                "patient_id": r.patient_id,
                "study_id": r.study_id,
                "width": r.width,
                "height": r.height,
                "channels": r.channels,
                "labels": ",".join(r.labels),
                "fracture_boxes": len(r.fracture_boxes),
                "annotation_format": r.annotation_format or "",
                "unreadable": r.unreadable,
            }
            for r in records
        ],
    )
    elapsed = _now() - started
    _safe_print(
        "inspect complete: "
        f"images={len(records)} "
        f"elapsed={elapsed:.1f}s "
        f"missing_images={summary['missing_images']} "
        f"invalid_images={summary['invalid_images']} "
        f"duplicate_groups={summary['duplicate_groups']}"
    )
    _safe_print(f"stage timing: {report['profiling']['stage_times']}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    started = _now()
    _safe_print(
        "convert: "
        f"workers={args.workers} "
        f"io_workers={args.io_workers} "
        f"batch_size={args.batch_size} "
        f"progress={args.progress}"
    )
    records, summary = build_records(
        args.raw_dir,
        workers=args.io_workers,
        hash_workers=args.hash_workers,
        batch_size=args.batch_size,
        progress=args.progress,
        force=args.force,
    )
    comparison = summary.get("comparison", compare_annotation_formats(args.raw_dir))
    split_records = make_splits(records, seed=args.seed)
    conversion = convert_to_yolo(
        records,
        args.processed_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        progress=args.progress,
        force=args.force,
    )
    split_stats = save_split_files(split_records, args.splits_dir)
    save_json(
        args.conversion_report,
        {
            "conversion": conversion,
            "comparison": comparison,
            "profiling": summary.get("stage_times", {}),
        },
    )
    save_json(args.split_report, split_stats)
    save_json(args.dataset_report, summarize(records, comparison))
    generate_dataset_figures(records, args.figures_dir, force=args.force)
    build_final_dataset(
        records,
        args.processed_dir,
        split_records,
        force=args.force,
        workers=args.io_workers,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    _safe_print(
        _stage_status(
            "convert",
            len(records),
            len(records),
            started,
            args.workers,
            conversion.get("copy_strategy", ""),
        )
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    return cmd_convert(args)


def cmd_validate(args: argparse.Namespace) -> int:
    records, summary = build_records(
        args.raw_dir,
        workers=args.io_workers,
        batch_size=args.batch_size,
        progress=args.progress,
        force=args.force,
    )
    split_records = make_splits(records, seed=args.seed)
    save_json(
        args.validation_report,
        {
            "raw": validate_records(records, split_records),
            "processed": validate_processed_dataset(
                args.processed_dir,
                workers=args.io_workers,
                batch_size=args.batch_size,
                progress=args.progress,
            ),
            "inspection": summary.get("stage_times", {}),
        },
    )
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    save_json(
        args.smoke_report,
        smoke_load_dataset(args.processed_dir / "yolo", max_batches=args.max_batches),
    )
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    records, _ = build_records(
        args.raw_dir,
        workers=args.io_workers,
        batch_size=args.batch_size,
        progress=args.progress,
        force=args.force,
    )
    split_records = make_splits(records, seed=args.seed)
    save_json(args.split_report, save_split_files(split_records, args.splits_dir))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    records, _ = build_records(
        args.raw_dir,
        workers=args.io_workers,
        batch_size=args.batch_size,
        progress=args.progress,
        force=args.force,
    )
    generate_dataset_figures(records, args.figures_dir, force=args.force)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset preparation workflow")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download")
    d.add_argument("--output-dir", type=Path, default=get_paths().raw / "archives")
    d.add_argument(
        "--manifest", type=Path, default=get_paths().dataset_reports / "download_manifest.json"
    )
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    e = sub.add_parser("extract")
    e.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    e.add_argument(
        "--manifest", type=Path, default=get_paths().dataset_reports / "extraction_manifest.json"
    )
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_extract)

    i = sub.add_parser("inspect")
    i.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    i.add_argument(
        "--report-json", type=Path, default=get_paths().dataset_reports / "dataset_report.json"
    )
    i.add_argument(
        "--report-csv", type=Path, default=get_paths().dataset_reports / "dataset_report.csv"
    )
    i.add_argument("--workers", type=int, default=_default_workers())
    i.add_argument("--hash-workers", type=int, default=None)
    i.add_argument("--batch-size", type=int, default=16)
    i.add_argument("--force", action="store_true")
    i.add_argument("--progress", action="store_true")
    i.set_defaults(func=cmd_inspect)

    c = sub.add_parser("convert")
    c.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    c.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    c.add_argument("--splits-dir", type=Path, default=get_paths().splits)
    c.add_argument(
        "--dataset-report", type=Path, default=get_paths().dataset_reports / "dataset_report.json"
    )
    c.add_argument(
        "--conversion-report",
        type=Path,
        default=get_paths().dataset_reports / "conversion_report.json",
    )
    c.add_argument(
        "--split-report", type=Path, default=get_paths().dataset_reports / "split_report.json"
    )
    c.add_argument("--figures-dir", type=Path, default=get_paths().figures / "dataset_statistics")
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--workers", type=int, default=_default_workers())
    c.add_argument("--io-workers", type=int, default=max(8, _default_workers() * 2))
    c.add_argument("--hash-workers", type=int, default=None)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--force", action="store_true")
    c.add_argument("--progress", action="store_true")
    c.set_defaults(func=cmd_convert)

    sp = sub.add_parser("split")
    sp.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    sp.add_argument("--splits-dir", type=Path, default=get_paths().splits)
    sp.add_argument(
        "--split-report", type=Path, default=get_paths().dataset_reports / "split_report.json"
    )
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--io-workers", type=int, default=max(8, _default_workers() * 2))
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--progress", action="store_true")
    sp.set_defaults(func=cmd_split)

    f = sub.add_parser("figures")
    f.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    f.add_argument("--figures-dir", type=Path, default=get_paths().figures / "dataset_statistics")
    f.add_argument("--io-workers", type=int, default=max(8, _default_workers() * 2))
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--force", action="store_true")
    f.add_argument("--progress", action="store_true")
    f.set_defaults(func=cmd_figures)

    v = sub.add_parser("validate")
    v.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    v.add_argument(
        "--validation-report",
        type=Path,
        default=get_paths().dataset_reports / "validation_report.json",
    )
    v.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--io-workers", type=int, default=max(8, _default_workers() * 2))
    v.add_argument("--batch-size", type=int, default=64)
    v.add_argument("--force", action="store_true")
    v.add_argument("--progress", action="store_true")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("smoke")
    s.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    s.add_argument(
        "--smoke-report",
        type=Path,
        default=get_paths().dataset_reports / "loader_smoke_report.json",
    )
    s.add_argument("--max-batches", type=int, default=2)
    s.set_defaults(func=cmd_smoke)

    p2 = sub.add_parser("prepare")
    p2.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    p2.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    p2.add_argument("--splits-dir", type=Path, default=get_paths().splits)
    p2.add_argument(
        "--dataset-report", type=Path, default=get_paths().dataset_reports / "dataset_report.json"
    )
    p2.add_argument(
        "--conversion-report",
        type=Path,
        default=get_paths().dataset_reports / "conversion_report.json",
    )
    p2.add_argument(
        "--split-report", type=Path, default=get_paths().dataset_reports / "split_report.json"
    )
    p2.add_argument("--figures-dir", type=Path, default=get_paths().figures / "dataset_statistics")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--workers", type=int, default=_default_workers())
    p2.add_argument("--io-workers", type=int, default=max(8, _default_workers() * 2))
    p2.add_argument("--hash-workers", type=int, default=None)
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--force", action="store_true")
    p2.add_argument("--progress", action="store_true")
    p2.set_defaults(func=cmd_prepare)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
