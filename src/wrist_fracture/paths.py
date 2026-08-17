from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    splits: Path
    outputs: Path
    audits: Path
    dataset_reports: Path
    experiments: Path
    figures: Path
    tables: Path

    def ensure(self) -> ProjectPaths:
        for path in [
            self.data,
            self.raw,
            self.interim,
            self.processed,
            self.splits,
            self.outputs,
            self.audits,
            self.dataset_reports,
            self.experiments,
            self.figures,
            self.tables,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths(root: Path | None = None) -> ProjectPaths:
    root = root or get_project_root()
    data = root / "data"
    outputs = root / "outputs"
    return ProjectPaths(
        root=root,
        data=data,
        raw=data / "raw",
        interim=data / "interim",
        processed=data / "processed",
        splits=data / "splits",
        outputs=outputs,
        audits=outputs / "audits",
        dataset_reports=outputs / "dataset_reports",
        experiments=outputs / "experiments",
        figures=outputs / "figures",
        tables=outputs / "tables",
    )
