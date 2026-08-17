from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def git_status(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "status", "--short"], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def git_dirty(root: Path) -> bool:
    status = git_status(root)
    return bool(status)


def dependency_lock_hash(root: Path) -> str | None:
    lock = root / "uv.lock"
    return sha256_file(lock) if lock.exists() else None


@dataclass(frozen=True)
class EnvironmentReport:
    timestamp_utc: str
    python_version: str
    python_executable: str
    platform: str
    cpu: str
    ram_gb: float
    disk_free_gb: float
    torch_version: str | None
    torch_cuda: str | None
    cuda_available: bool | None
    cudnn_version: int | None
    gpu: list[dict[str, Any]]
    ultralytics_version: str | None
    compute_capability: list[str]
    amp_support: bool | None
    bf16_support: bool | None
    nvidia_smi: str | None


def collect_environment_report(root: Path) -> EnvironmentReport:
    gpu: list[dict[str, Any]] = []
    compute_capability: list[str] = []
    torch_version = torch_cuda = ultralytics_version = None
    cuda_available = None
    cudnn_version = None
    amp_support = bf16_support = None
    nvidia_smi = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        cudnn_version = torch.backends.cudnn.version()
        amp_support = bool(cuda_available and torch.cuda.is_available())
        bf16_support = bool(
            cuda_available
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                cc = f"{p.major}.{p.minor}"
                compute_capability.append(cc)
                gpu.append(
                    {
                        "index": i,
                        "name": p.name,
                        "total_memory_gb": round(p.total_memory / 1024**3, 2),
                        "capability": cc,
                    }
                )
    except Exception:
        pass
    try:
        nvidia_smi = (
            subprocess.run(
                ["nvidia-smi"], cwd=root, text=True, capture_output=True, check=False
            ).stdout.strip()
            or None
        )
    except Exception:
        nvidia_smi = None
    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except Exception:
        pass
    return EnvironmentReport(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 2),
        disk_free_gb=round(psutil.disk_usage(str(root)).free / 1024**3, 2),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        cudnn_version=cudnn_version,
        gpu=gpu,
        ultralytics_version=ultralytics_version,
        compute_capability=compute_capability,
        amp_support=amp_support,
        bf16_support=bf16_support,
        nvidia_smi=nvidia_smi,
    )


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
