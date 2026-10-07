"""Provider adapter interface."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from core.models import ProviderResponse


class Provider(ABC):
    name: str

    def __init__(self, name: str, settings: dict[str, Any]):
        self.name = name
        self.settings = settings
        self.model = settings.get("model", "")
        self.temperature = settings.get("temperature")
        self.max_output_tokens = settings.get("max_output_tokens", 1800)

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str, *, case_id: str,
                 json_only: bool = False) -> ProviderResponse:
        """Single-turn completion. json_only asks for a bare JSON object (used by the judge)."""

    @staticmethod
    def _ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
