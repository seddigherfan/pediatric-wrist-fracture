from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import psutil

from wrist_fracture.provenance import collect_environment_report


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=False)
        text = (result.stdout or result.stderr).strip()
        return text or None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--dataset-yaml", default="data/processed/yolo/dataset.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    root = Path.cwd()
    env = collect_environment_report(root)
    try:
        import torch

        torch_version = torch.__version__
        cuda_build = torch.version.cuda is not None
        cuda_available = torch.cuda.is_available()
        cudnn = torch.backends.cudnn.version()
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            total_vram_gb = round(props.total_memory / 1024**3, 2)
            free_vram_gb = round(
                (
                    torch.cuda.mem_get_info(0)[0]
                    if hasattr(torch.cuda, "mem_get_info")
                    else props.total_memory
                )
                / 1024**3,
                2,
            )
            capability = f"{props.major}.{props.minor}"
            amp_support = True
            bf16_support = bool(
                hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            )
            gpu_count = torch.cuda.device_count()
        else:
            gpu_name = None
            total_vram_gb = None
            free_vram_gb = None
            capability = None
            amp_support = False
            bf16_support = False
            gpu_count = 0
    except Exception:
        torch_version = None
        cuda_build = None
        cuda_available = None
        cudnn = None
        gpu_name = None
        total_vram_gb = None
        free_vram_gb = None
        capability = None
        amp_support = None
        bf16_support = None
        gpu_count = None
    dataset_yaml = root / args.dataset_yaml
    output_dir = root / args.output_dir
    output_writeable = True
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception:
        output_writeable = False
    payload = {
        "nvidia_smi": _run(["nvidia-smi"]),
        "driver_version": env.nvidia_smi.splitlines()[2] if env.nvidia_smi else None,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb,
        "torch_version": torch_version,
        "torch_cuda_build": cuda_build,
        "torch_cuda_version": env.torch_cuda,
        "cudnn_version": cudnn,
        "compute_capability": capability,
        "amp_support": amp_support,
        "bf16_support": bf16_support,
        "ultralytics_version": env.ultralytics_version,
        "dataset_accessible": dataset_yaml.exists(),
        "output_directory_writeable": output_writeable,
        "free_disk_gb": round(psutil.disk_usage(str(root)).free / 1024**3, 2),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_gpu and not cuda_available:
        raise SystemExit("GPU required but not available on this machine.")


if __name__ == "__main__":
    main()
