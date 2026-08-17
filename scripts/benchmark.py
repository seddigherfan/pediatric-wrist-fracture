from __future__ import annotations

import argparse
from pathlib import Path

from wrist_fracture.config import ConfigError, load_config_bundle, validate_experiment_config
from wrist_fracture.validation_benchmark_suite import (
    benchmark_checkpoint,
    build_benchmark_image_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark-id")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not args.execute and not args.dry_run:
        raise ConfigError("benchmark requires explicit --execute")
    cfg = load_config_bundle(
        args.config,
        model_path=args.model_config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )
    errors = validate_experiment_config(cfg, dry_run=not args.execute)
    if errors:
        raise ConfigError("; ".join(errors))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists() or not checkpoint.is_file():
        raise ConfigError(f"checkpoint not found: {checkpoint}")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/benchmarks") / (args.benchmark_id or checkpoint.stem)
    ).resolve()
    print(
        {
            "config": args.config,
            "checkpoint": str(checkpoint),
            "warmup": args.warmup,
            "samples": args.samples,
            "batch_size": args.batch_size,
            "device": args.device,
            "output_dir": str(out_dir),
        }
    )
    if args.dry_run or not args.execute:
        return
    if out_dir.exists():
        raise ConfigError(f"output collision: {out_dir}")
    manifest = build_benchmark_image_manifest(
        Path("data/processed/yolo/images"), split="val", samples=args.samples
    )
    images = [Path("data/processed/yolo/images") / rel for rel in manifest["selected_samples"]]
    benchmark_checkpoint(
        checkpoint=checkpoint,
        images=images,
        cfg_path=Path(args.config),
        model_cfg=Path(args.model_config) if args.model_config else None,
        hardware_cfg=Path(args.hardware_config) if args.hardware_config else None,
        run_cfg=Path(args.run_config) if args.run_config else None,
        warmup=args.warmup,
        samples=args.samples,
        batch_size=args.batch_size,
        execute=True,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
