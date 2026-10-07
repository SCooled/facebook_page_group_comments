from __future__ import annotations

import os
import time

from openai import OpenAI

from core.models import ProviderResponse
from providers.base import Provider


class OpenAIProvider(Provider):
    """Chat Completions. Also used for any OpenAI-compatible endpoint via base_url."""

    def __init__(self, name, settings):
        super().__init__(name, settings)
        api_key = os.environ.get(settings.get("api_key_env", "OPENAI_API_KEY"))
        self.client = OpenAI(api_key=api_key, base_url=settings.get("base_url"))

    def complete(self, system_prompt, user_message, *, case_id, json_only=False):
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=self.max_output_tokens,
        )
        if json_only:
            kwargs["response_format"] = {"type": "json_object"}
        effective_temp = None
        start = time.perf_counter()
        try:
            if self.temperature is not None:
                resp = self.client.chat.completions.create(temperature=self.temperature, **kwargs)
                effective_temp = self.temperature
            else:
                resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # some models reject temperature; retry without it
            if self.temperature is not None and "temperature" in str(e):
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        usage = resp.usage
        return ProviderResponse(
            text=resp.choices[0].message.content or "",
            model_name=resp.model or self.model,
            latency_ms=self._ms(start),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            effective_temperature=effective_temp,
        )
