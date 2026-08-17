from __future__ import annotations

import sys
from pathlib import Path

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.download_dataset import inspect_dataset_record


def main() -> None:
    record = inspect_dataset_record()
    print(record["article"].get("title"))
    print(f"files={len(record['files'])}")


if __name__ == "__main__":
    main()
