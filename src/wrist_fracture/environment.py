from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from .paths import get_paths


@dataclass(frozen=True)
class EnvironmentMetadata:
    timestamp_utc: str
    git_commit: str | None
    python_version: str
    python_executable: str
    os: str
    cpu: str
    ram_gb: float
    disk_free_gb: float
    gpu: list[dict[str, Any]]
    cuda_available: bool | None
    torch_version: str | None
    torch_cuda: str | None


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_metadata(root: Path | None = None) -> EnvironmentMetadata:
    root = root or get_paths().root
    gpu: list[dict[str, Any]] = []
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpu.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    }
                )
    except Exception:
        torch_version = None
        torch_cuda = None
        cuda_available = None

    return EnvironmentMetadata(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        git_commit=_run_git(["rev-parse", "HEAD"], root),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 2),
        disk_free_gb=round(psutil.disk_usage(str(root)).free / 1024**3, 2),
        gpu=gpu,
        cuda_available=cuda_available,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
    )


def main() -> None:
    print(json.dumps(asdict(collect_environment_metadata()), indent=2, sort_keys=True))
