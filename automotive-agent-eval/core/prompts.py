"""System prompt assembly."""
from __future__ import annotations

import hashlib
from pathlib import Path

OUTPUT_FORMAT_FILE = "output_format.txt"


def build_system_prompt(prompts_dir: Path, files: list[str]) -> str:
    chunks = [(prompts_dir / f).read_text().strip() for f in files]
    chunks.append((prompts_dir / OUTPUT_FORMAT_FILE).read_text().strip())
    return "\n\n".join(chunks) + "\n"


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_prompt(prompts_dir: Path, name: str) -> str:
    return (prompts_dir / name).read_text()
