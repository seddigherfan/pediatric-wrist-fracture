from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from wrist_fracture.calibration import (
    CALIBRATED_MODEL_ORDER,
    build_calibration_report,
    build_command,
    build_hardware_profile,
    build_recommended_full_config,
    build_resume_state,
    candidate_batches,
    discover_latest_completed_execution,
    flatten_candidate_rows,
    load_calibration_report,
    now_utc,
    recover_legacy_execution,
    report_has_complete_real_evidence,
    update_run_config_batch,
    write_atomic,
    write_execution_artifacts,
    write_reports,
    write_resume_state,
)
from wrist_fracture.calibration_probe import run_bounded_training_probe
from wrist_fracture.config import ConfigError, load_config_bundle
from wrist_fracture.provenance import collect_environment_report, git_commit, git_dirty, to_jsonable


def _default_output_dir(root: Path) -> Path:
    return root / "outputs" / "calibration"


def _load_resume_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        exec_root = discover_latest_completed_execution(path.parent)
        if exec_root is not None:
            state_path = exec_root / "resume_state.json"
            if state_path.exists():
                return json.loads(state_path.read_text(encoding="utf-8"))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_candidate_batches(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parsed = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return candidate_batches(candidates=parsed)


def _resume_state_for_model(path: Path, model_family: str) -> dict[str, Any] | None:
    data = _load_resume_state(path)
    if not data:
        return None
    if data.get("model_family") in {None, model_family, "all"}:
        return data
    return None


def _candidate_plan(
    *,
    cfg,
    model_family: str,
    batches: list[int],
    out_dir: Path,
    resume_state: dict[str, Any] | None,
    force: bool,
) -> list[dict[str, Any]]:
    completed = {int(row["batch_size"]) for row in (resume_state or {}).get("results", [])}
    rows: list[dict[str, Any]] = []
    seen_oom = False
    for batch in batches:
        if seen_oom:
            rows.append(
                {
                    "model_family": model_family,
                    "batch_size": batch,
                    "status": "skipped_after_oom",
                    "status_detail": "skipped after OOM on smaller batch",
                }
            )
            continue
        if batch in completed and not force:
            rows.append(
                {
                    "model_family": model_family,
                    "batch_size": batch,
                    "status": "skipped",
                    "status_detail": "already completed",
                }
            )
            continue
        probe_cfg = dataclasses.replace(cfg, batch_size=batch)
        result = run_bounded_training_probe(probe_cfg, out_dir=out_dir, iterations=30)
        rows.append(result)
        if result.get("status") == "oom":
            seen_oom = True
    return rows


def _plan_only_rows(target_models: list[str], batches: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "model_family": model_family,
            "batch_size": batch,
            "status": "planned",
            "status_detail": "dry-run only",
        }
        for model_family in target_models
        for batch in batches
    ]


def _selected_common_batch(
    results_by_model: dict[str, list[dict[str, Any]]], fallback: int
) -> int | None:
    largest_stables = []
    for rows in results_by_model.values():
        stable = [int(row["batch_size"]) for row in rows if row.get("status") == "stable"]
        if not stable:
            return None
        largest_stables.append(max(stable))
    return min(largest_stables) if largest_stables else None


def _execution_persistence_root(out_dir: Path) -> Path:
    return out_dir / "executions"


def _load_effective_evidence(out_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    recovered = discover_latest_completed_execution(out_dir) or recover_legacy_execution(out_dir)
    if recovered is None:
        return None, None
    report = load_calibration_report(recovered / "calibration_report.json")
    if not report_has_complete_real_evidence(report or {}):
        return None, None
    return recovered, report


def run(args: argparse.Namespace) -> int:
    root = Path.cwd()
    base_cfg = load_config_bundle(
        args.config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_state_path = out_dir / "resume_state.json"
    batches = (
        _parse_candidate_batches(getattr(args, "candidate_batches", None)) or candidate_batches()
    )
    target_models = [base_cfg.model.family] if args.model_config else list(CALIBRATED_MODEL_ORDER)
    if args.apply and not args.execute:
        evidence_root, evidence = _load_effective_evidence(out_dir)
        if evidence_root is None or evidence is None:
            raise ConfigError("apply requires complete real calibration evidence")
        if not report_has_complete_real_evidence(evidence):
            raise ConfigError("apply requires complete real calibration evidence")
        selected_batch = evidence.get("common_stable_batch")
        if selected_batch is None:
            raise ConfigError("apply requires a selected common stable batch")
        update_run_config_batch(root / "configs" / "runs" / "full.yaml", batch_size=selected_batch)
        applied_cfg = load_config_bundle(
            args.config,
            model_path=Path(args.model_config or f"configs/models/{target_models[0]}.yaml"),
            hardware_path=args.hardware_config,
            run_path=root / "configs" / "runs" / "full.yaml",
        )
        if applied_cfg.batch_size != selected_batch:
            raise ConfigError("applied calibration batch did not propagate into ExperimentConfig")
        write_atomic(
            evidence_root / "application_record.json",
            json.dumps(
                {
                    "execution_id": evidence.get("execution_id") or evidence_root.name,
                    "report_sha256": evidence.get("report_sha256"),
                    "selected_batch": selected_batch,
                    "applied_batch": applied_cfg.batch_size,
                    "timestamp_utc": now_utc(),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return 0
    if not args.execute and not args.apply:
        candidate_rows = _plan_only_rows(target_models, batches)
        report = build_calibration_report(
            cfg=base_cfg,
            model_family="all" if len(target_models) > 1 else target_models[0],
            candidates=candidate_rows,
            recommended_batch_size=None,
            applied=False,
            resume_state=build_resume_state(
                model_family="all" if len(target_models) > 1 else target_models[0],
                candidate_order=batches,
                results=candidate_rows,
            ),
        )
        hardware_profile = build_hardware_profile(base_cfg)
        environment = {
            "timestamp_utc": now_utc(),
            "git_commit": git_commit(root),
            "git_dirty": git_dirty(root),
            "environment": to_jsonable(collect_environment_report(root)),
        }
        commands = [
            build_command(
                model_path=Path(args.model_config or f"configs/models/{target_models[0]}.yaml"),
                hardware_path=Path(args.hardware_config or "configs/hardware/rtx4090.yaml"),
                run_path=Path(args.run_config or "configs/runs/full.yaml"),
                execute=False,
            )
        ]
        recommended_config = build_recommended_full_config(base_cfg, batch_size=base_cfg.batch_size)
        write_atomic(
            out_dir / "calibration_plan.json",
            json.dumps(report, indent=2, sort_keys=True),
        )
        write_atomic(
            out_dir / "calibration_plan.csv",
            "\n".join([",".join(map(str, row.values())) for row in candidate_rows]) + "\n",
        )
        if not report_has_complete_real_evidence(
            load_calibration_report(out_dir / "calibration_report.json") or {}
        ):
            write_reports(
                out_dir,
                report=report,
                hardware_profile=hardware_profile,
                stage_timings={
                    "planning_seconds": 0.0,
                    "selection_seconds": 0.0,
                    "reporting_seconds": 0.0,
                },
                environment=environment,
                commands=commands,
                recommended_config=recommended_config,
            )
        if args.print_report:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    per_model_rows: dict[str, list[dict[str, Any]]] = {}
    if args.execute:
        for model_family in target_models:
            model_cfg = load_config_bundle(
                args.config,
                model_path=Path(args.model_config)
                if args.model_config
                else Path(f"configs/models/{model_family}.yaml"),
                hardware_path=args.hardware_config,
                run_path=args.run_config,
            )
            model_resume = (
                _resume_state_for_model(resume_state_path, model_family) if args.resume else None
            )
            candidate_rows = _candidate_plan(
                cfg=model_cfg,
                model_family=model_family,
                batches=batches,
                out_dir=out_dir,
                resume_state=model_resume,
                force=bool(args.force),
            )
            per_model_rows[model_family] = candidate_rows
        recommended_batch = _selected_common_batch(per_model_rows, base_cfg.batch_size)
        candidate_rows = flatten_candidate_rows(per_model_rows)
    report = build_calibration_report(
        cfg=base_cfg,
        model_family="all" if len(target_models) > 1 else target_models[0],
        candidates=candidate_rows,
        recommended_batch_size=recommended_batch,
        applied=bool(args.apply),
        resume_state=build_resume_state(
            model_family="all" if len(target_models) > 1 else target_models[0],
            candidate_order=batches,
            results=candidate_rows,
        ),
    )
    hardware_profile = build_hardware_profile(base_cfg)
    stage_timings = {
        "planning_seconds": 0.0,
        "selection_seconds": 0.0,
        "reporting_seconds": 0.0,
    }
    environment = {
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "environment": to_jsonable(collect_environment_report(root)),
    }
    commands = [
        build_command(
            model_path=Path(args.model_config or f"configs/models/{target_models[0]}.yaml"),
            hardware_path=Path(args.hardware_config or "configs/hardware/rtx4090.yaml"),
            run_path=Path(args.run_config or "configs/runs/full.yaml"),
            execute=True,
        ),
    ]
    recommended_config = build_recommended_full_config(
        base_cfg, batch_size=recommended_batch or base_cfg.batch_size
    )
    execution_id = (
        report.get("execution_id") or f"calib-{now_utc().replace(':', '').replace('-', '')}"
    )
    execution_root = write_execution_artifacts(
        out_dir,
        execution_id=execution_id,
        report=report,
        hardware_profile=hardware_profile,
        stage_timings=stage_timings,
        environment=environment,
        commands=commands,
        recommended_config=recommended_config,
    )
    write_resume_state(
        execution_root,
        build_resume_state(
            model_family=("all" if len(target_models) > 1 else target_models[0]),
            candidate_order=batches,
            results=candidate_rows,
        ),
    )
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.apply:
        evidence_root, evidence = _load_effective_evidence(out_dir)
        if evidence_root is None or evidence is None:
            raise ConfigError("apply requires complete real calibration evidence")
        if not report_has_complete_real_evidence(evidence):
            raise ConfigError("apply requires complete real calibration evidence")
        selected_batch = evidence.get("common_stable_batch")
        if selected_batch is None:
            raise ConfigError("apply requires a selected common stable batch")
        update_run_config_batch(root / "configs" / "runs" / "full.yaml", batch_size=selected_batch)
        applied_cfg = load_config_bundle(
            args.config,
            model_path=Path(args.model_config or f"configs/models/{target_models[0]}.yaml"),
            hardware_path=args.hardware_config,
            run_path=root / "configs" / "runs" / "full.yaml",
        )
        if applied_cfg.batch_size != selected_batch:
            raise ConfigError("applied calibration batch did not propagate into ExperimentConfig")
        write_atomic(
            evidence_root / "application_record.json",
            json.dumps(
                {
                    "execution_id": evidence.get("execution_id") or evidence_root.name,
                    "report_sha256": evidence.get("report_sha256"),
                    "selected_batch": selected_batch,
                    "applied_batch": applied_cfg.batch_size,
                    "timestamp_utc": now_utc(),
                },
                indent=2,
                sort_keys=True,
            ),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--print-report", action="store_true")
    parser.add_argument("--candidate-batches")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        raise ConfigError("--execute and --dry-run are mutually exclusive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
