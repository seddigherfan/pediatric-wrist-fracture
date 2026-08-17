from __future__ import annotations

import argparse
from pathlib import Path

from wrist_fracture.config import ConfigError, load_config_bundle, validate_experiment_config
from wrist_fracture.validation_benchmark_suite import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if not args.execute and not args.dry_run and not args.preflight:
        raise ConfigError("evaluation requires explicit --execute")
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
        else Path("outputs/evaluations") / (args.evaluation_id or checkpoint.stem)
    ).resolve()
    plan = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "output_dir": str(out_dir),
        "execute": args.execute,
        "dry_run": args.dry_run,
    }
    print(plan)
    if args.dry_run or args.preflight or not args.execute:
        return
    evaluate_checkpoint(
        checkpoint=checkpoint,
        cfg_path=Path(args.config),
        model_cfg=Path(args.model_config) if args.model_config else None,
        hardware_cfg=Path(args.hardware_config) if args.hardware_config else None,
        run_cfg=Path(args.run_config) if args.run_config else None,
        split=args.split,
        execute=True,
        out_dir=out_dir,
        allow_test=args.allow_test,
    )


if __name__ == "__main__":
    raise SystemExit(main())
