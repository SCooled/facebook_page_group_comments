"""Placeholder Instinct adapter.

The Instinct API has not been specified. Until it is, this adapter assumes an
OpenAI-compatible chat endpoint reachable at settings.base_url with INSTINCT_API_KEY.
If base_url is not configured it raises so runs fail loudly rather than silently
producing empty results.
"""
from __future__ import annotations

from providers.base import Provider
from providers.openai_adapter import OpenAIProvider


class InstinctProvider(Provider):
    def __init__(self, name, settings):
        super().__init__(name, settings)
        if not settings.get("base_url"):
            raise NotImplementedError(
                "Instinct adapter is a placeholder: set providers.instinct.base_url in config.yaml "
                "(OpenAI-compatible endpoint assumed) and INSTINCT_API_KEY in the environment."
            )
        self._inner = OpenAIProvider(name, {**settings, "api_key_env": settings.get("api_key_env", "INSTINCT_API_KEY")})

    def complete(self, system_prompt, user_message, *, case_id, json_only=False):
        return self._inner.complete(system_prompt, user_message, case_id=case_id, json_only=json_only)
