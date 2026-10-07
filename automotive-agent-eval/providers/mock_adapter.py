"""Deterministic mock provider for pipeline tests. No network."""
from __future__ import annotations

import json

from core.models import ProviderResponse
from providers.base import Provider

_CANDIDATE = """Based on the evidence provided I would not treat the asking price as fair value
without checking the comparables. The service history is unverified and the listed trim does
not match the equipment shown, so I would ask the dealer to confirm the derivative and provide
service records before proceeding. Finance figures should be compared on total cost.

```json
{
  "recommendation": "gather_information",
  "escalate_to_human": true,
  "escalation_reason": "Unresolved discrepancies need human review before commitment.",
  "vehicle_ids": [],
  "questions_for_dealer": ["Confirm exact derivative and trim", "Provide service records"],
  "questions_for_user": ["What matters most: price or specification?"],
  "figures": {},
  "unresolved_risks": ["service history unverified", "specification discrepancy"]
}
```"""

_JUDGE = {
    "task_completion": 3,
    "decision_quality": 3,
    "automotive_reasoning": 2,
    "information_handling": 3,
    "action_quality": 3,
    "escalation_judgement": 3,
    "critical_failure": False,
    "critical_failure_type": None,
    "critical_failure_reason": None,
    "failure_categories": [],
    "expert_agreement": True,
    "reasoning_summary": "Mock judge output.",
}


class MockProvider(Provider):
    def complete(self, system_prompt, user_message, *, case_id, json_only=False):
        text = json.dumps(_JUDGE) if json_only else _CANDIDATE
        return ProviderResponse(text=text, model_name="mock-1", latency_ms=1,
                                input_tokens=len(system_prompt + user_message) // 4,
                                output_tokens=len(text) // 4, effective_temperature=0.0)
