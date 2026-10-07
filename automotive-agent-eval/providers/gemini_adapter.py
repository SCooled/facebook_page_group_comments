from __future__ import annotations

import os
import time

from google import genai
from google.genai import types

from core.models import ProviderResponse
from providers.base import Provider


class GeminiProvider(Provider):
    def __init__(self, name, settings):
        super().__init__(name, settings)
        api_key = os.environ.get(settings.get("api_key_env", "GEMINI_API_KEY")) or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)

    def complete(self, system_prompt, user_message, *, case_id, json_only=False):
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_only else None,
        )
        start = time.perf_counter()
        resp = self.client.models.generate_content(model=self.model, contents=user_message, config=config)
        usage = resp.usage_metadata
        return ProviderResponse(
            text=resp.text or "",
            model_name=self.model,
            latency_ms=self._ms(start),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            effective_temperature=self.temperature,
        )
