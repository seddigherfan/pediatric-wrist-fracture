from __future__ import annotations

import sys
from pathlib import Path

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wrist_fracture.environment import collect_environment_metadata
from wrist_fracture.paths import get_paths


def main() -> None:
    root = get_paths().root
    print(collect_environment_metadata(root))


if __name__ == "__main__":
    main()
