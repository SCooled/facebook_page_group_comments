"""Condition C: the human expert's written answer, read from human_responses/<case_id>.md.

The human answer is scored by the same judge, blind, as any other provider.
"""
from __future__ import annotations

from pathlib import Path

from core.models import ProviderResponse
from providers.base import Provider


class HumanProvider(Provider):
    def __init__(self, name, settings):
        super().__init__(name, settings)
        self.dir = Path(settings["responses_dir"])
        self.model = "human_expert"

    def complete(self, system_prompt, user_message, *, case_id, json_only=False):
        path = self.dir / f"{case_id}.md"
        if not path.exists():
            raise FileNotFoundError(f"no human response for {case_id} at {path}")
        return ProviderResponse(text=path.read_text(), model_name="human_expert", latency_ms=0)
