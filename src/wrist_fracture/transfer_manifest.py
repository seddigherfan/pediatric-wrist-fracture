from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from wrist_fracture.provenance import dependency_lock_hash, git_commit, git_dirty, sha256_file


def _file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = json.dumps(
        sorted(
            (
                str(item.relative_to(path)).replace("\\", "/"),
                sha256_file(item),
            )
            for item in path.rglob("*")
            if item.is_file()
        ),
        separators=(",", ":"),
    )
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def _read_patient_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [str(row["patient_id"]) for row in rows if row.get("patient_id") not in (None, "")]


def _ensure_patient_manifest(root: Path, split: str) -> Path:
    csv_path = root / "data/splits" / f"{split}_patients.csv"
    txt_path = root / "data/splits" / f"{split}_patients.txt"
    patients = _read_patient_csv(csv_path)
    content = "\n".join(patients) + ("\n" if patients else "")
    if not txt_path.exists() or txt_path.read_text(encoding="utf-8") != content:
        with txt_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
    return txt_path


def _manifest_hash(root: Path, split: str) -> str | None:
    txt_path = _ensure_patient_manifest(root, split)
    return _file_hash(txt_path)


def _load_audit(root: Path) -> dict[str, object]:
    audit_path = root / "outputs/dataset_reports/final_dataset_audit.json"
    if not audit_path.exists():
        return {}
    return json.loads(audit_path.read_text(encoding="utf-8"))


def _expected_counts(root: Path) -> dict[str, int] | None:
    audit = _load_audit(root)
    labels = audit.get("labels")
    splits = audit.get("splits")
    if not isinstance(labels, dict) or not isinstance(splits, dict):
        return None
    totals = splits.get("totals")
    if not isinstance(totals, dict):
        return None
    images = labels.get("label_files")
    patients = totals.get("patient_count")
    if not all(isinstance(v, int) for v in (images, patients)):
        return None
    return {"images": int(images), "labels": int(images), "patients": int(patients)}


def _approximate_disk_requirement_gb(root: Path) -> float | None:
    yolo_dir = root / "data/processed/yolo"
    if not yolo_dir.exists():
        return None
    total_bytes = sum(item.stat().st_size for item in yolo_dir.rglob("*") if item.is_file())
    workspace_overhead_bytes = 1 * 1024**3
    return round((total_bytes + workspace_overhead_bytes) / 1_000_000_000, 2)


def build_manifest(root: Path, dataset_yaml: Path) -> dict[str, object]:
    return {
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "uv_lock_sha256": dependency_lock_hash(root),
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": _file_hash(dataset_yaml),
        "patient_split_hashes": {
            "train": _manifest_hash(root, "train"),
            "val": _manifest_hash(root, "val"),
            "test": _manifest_hash(root, "test"),
        },
        "image_split_hashes": {
            "train": _tree_hash(root / "data/processed/yolo/images/train"),
            "val": _tree_hash(root / "data/processed/yolo/images/val"),
            "test": _tree_hash(root / "data/processed/yolo/images/test"),
        },
        "final_dataset_audit_sha256": _file_hash(
            root / "outputs/dataset_reports/final_dataset_audit.json"
        ),
        "expected_counts": _expected_counts(root),
        "required_local_paths": [
            "data/processed/yolo/dataset.yaml",
            "outputs/dataset_reports/final_dataset_audit.json",
            "data/splits",
            "data/splits/train_patients.txt",
            "data/splits/val_patients.txt",
            "data/splits/test_patients.txt",
            "data/processed/yolo/images",
            "data/processed/yolo/labels",
        ],
        "approximate_disk_requirement_gb": _approximate_disk_requirement_gb(root),
    }


def verify_manifest(manifest: dict[str, object], root: Path, dataset_yaml: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("git_commit") != git_commit(root):
        errors.append("git commit mismatch")
    if manifest.get("git_dirty") != git_dirty(root):
        errors.append("dirty working-tree flag mismatch")
    if manifest.get("uv_lock_sha256") != dependency_lock_hash(root):
        errors.append("uv.lock hash mismatch")
    if manifest.get("dataset_yaml_sha256") != _file_hash(dataset_yaml):
        errors.append("dataset YAML hash mismatch")
    if manifest.get("final_dataset_audit_sha256") != _file_hash(
        root / "outputs/dataset_reports/final_dataset_audit.json"
    ):
        errors.append("dataset audit hash mismatch")
    if manifest.get("patient_split_hashes") != {
        "train": _manifest_hash(root, "train"),
        "val": _manifest_hash(root, "val"),
        "test": _manifest_hash(root, "test"),
    }:
        errors.append("patient split hash mismatch")
    if manifest.get("expected_counts") != _expected_counts(root):
        errors.append("expected counts mismatch")
    if manifest.get("approximate_disk_requirement_gb") != _approximate_disk_requirement_gb(root):
        errors.append("disk requirement mismatch")
    return errors
