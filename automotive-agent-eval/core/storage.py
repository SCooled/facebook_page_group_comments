"""JSON Lines persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel


def append_jsonl(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.model_dump_json() + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
