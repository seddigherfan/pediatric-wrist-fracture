# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wrist_fracture.data.preparation import save_json
from wrist_fracture.paths import get_paths

FIGSHARE_API = "https://api.figshare.com/v2/articles/14825193"


@dataclass(frozen=True)
class FigshareFile:
    id: int
    name: str
    download_url: str | None
    size: int | None
    checksum: str | None


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def inspect_dataset_record() -> dict[str, Any]:
    article = fetch_json(FIGSHARE_API)
    files: list[FigshareFile] = []
    for item in article.get("files", []):
        files.append(
            FigshareFile(
                id=int(item["id"]),
                name=str(item["name"]),
                download_url=item.get("download_url"),
                size=item.get("size"),
                checksum=item.get("supplied_md5") or item.get("computed_md5"),
            )
        )
    return {"article": article, "files": files}


def estimate_disk_requirement(files: list[FigshareFile]) -> int:
    return sum(file.size or 0 for file in files)


def checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".md5")


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_file(
    destination: Path, expected_size: int | None, expected_checksum: str | None
) -> tuple[bool, dict[str, Any]]:
    info = {"path": str(destination), "exists": destination.exists()}
    if not destination.exists():
        return False, info
    info["actual_size"] = destination.stat().st_size
    info["expected_size"] = expected_size
    info["actual_checksum"] = md5_file(destination)
    info["expected_checksum"] = expected_checksum
    if expected_size is not None and info["actual_size"] != expected_size:
        return False, info
    if expected_checksum and info["actual_checksum"] != expected_checksum:
        return False, info
    return True, info


def download_file(file: FigshareFile, target_dir: Path, force: bool = False) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    if not file.download_url:
        raise RuntimeError(f"No download URL available for {file.name}")
    destination = target_dir / file.name
    valid, info = is_valid_file(destination, file.size, file.checksum)
    if valid and not force:
        return {
            "file_name": file.name,
            "source_url": file.download_url,
            "expected_size": file.size,
            "actual_size": destination.stat().st_size,
            "expected_checksum": file.checksum,
            "actual_checksum": md5_file(destination),
            "verified": True,
            "local_path": str(destination),
            "downloaded": False,
        }
    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    headers = {}
    if destination.exists() and destination.stat().st_size and not force:
        headers["Range"] = f"bytes={destination.stat().st_size}-"
    req = urllib.request.Request(file.download_url, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total else None
        mode = "ab" if headers.get("Range") else "wb"
        with (
            tmp.open(mode) as out,
            tqdm(total=total_int, unit="B", unit_scale=True, desc=file.name) as bar,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    if destination.exists():
        destination.unlink()
    tmp.replace(destination)
    actual_size = destination.stat().st_size
    actual_checksum = md5_file(destination)
    verified = True
    if file.size is not None and actual_size != file.size:
        verified = False
    if file.checksum and actual_checksum != file.checksum:
        verified = False
    if file.checksum:
        checksum_path(destination).write_text(file.checksum, encoding="utf-8")
    return {
        "file_name": file.name,
        "source_url": file.download_url,
        "expected_size": file.size,
        "actual_size": actual_size,
        "expected_checksum": file.checksum,
        "actual_checksum": actual_checksum,
        "verified": verified,
        "local_path": str(destination),
        "downloaded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or download the GRAZPEDWRI-DX dataset.")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=get_paths().raw / "archives")
    parser.add_argument(
        "--manifest", type=Path, default=get_paths().dataset_reports / "download_manifest.json"
    )
    args = parser.parse_args()

    record = inspect_dataset_record()
    article = record["article"]
    files: list[FigshareFile] = record["files"]
    payload = {
        "title": article.get("title"),
        "doi": article.get("doi"),
        "id": article.get("id"),
        "published_date": article.get("published_date"),
        "file_count": len(files),
        "files": [file.__dict__ for file in files],
        "estimated_disk_bytes": estimate_disk_requirement(files),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.metadata_only:
        return 0
    manifest: list[dict[str, Any]] = []
    if args.dry_run:
        for file in files:
            manifest.append(
                {
                    "file_name": file.name,
                    "source_url": file.download_url,
                    "expected_size": file.size,
                    "expected_checksum": file.checksum,
                    "verified": False,
                    "local_path": str(args.output_dir / file.name),
                }
            )
    else:
        for file in files:
            manifest.append(download_file(file, args.output_dir, force=args.force))
    save_json(args.manifest, {"article": payload, "files": manifest})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
