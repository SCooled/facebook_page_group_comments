"""Extract the structured JSON block from a candidate's response."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from core.models import StructuredBlock

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_structured(text: str) -> tuple[StructuredBlock | None, str | None]:
    """Return (block, error). Takes the LAST fenced JSON object that validates."""
    candidates = _FENCE.findall(text)
    if not candidates:
        # Fall back to the last top-level {...} in the text.
        tail = text.rfind("{")
        if tail != -1:
            candidates = [text[tail:]]
    last_error = "no JSON block found"
    for raw in reversed(candidates):
        try:
            return StructuredBlock.model_validate(json.loads(raw)), None
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
    return None, last_error
