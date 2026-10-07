"""LLM judge. Blind to candidate provider and configuration."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from core.cases import build_candidate_payload
from core.models import Case, DeterministicResult, JudgeResult
from providers.base import Provider


def build_judge_prompt(template: str, case: Case, raw_response: str, det: list[DeterministicResult]) -> str:
    h = case.hidden
    det_lines = "\n".join(f"- [{'PASS' if d.passed else 'FAIL'}] {d.check_id}: {d.detail}" for d in det) or "- none"
    return (
        template.replace("{{CANDIDATE_INPUT}}", build_candidate_payload(case))
        .replace("{{CRITICAL_FACTS}}", _bullets(h.critical_facts))
        .replace("{{EXPECTED_BEHAVIOURS}}", _bullets(h.expected_behaviours))
        .replace("{{CRITICAL_FAILURES}}", _bullets(h.critical_failures))
        .replace("{{ESCALATION_EXPECTED}}", h.escalation_expected)
        .replace("{{EXPERT_NOTES}}", h.expert_notes or "none")
        .replace("{{DETERMINISTIC_RESULTS}}", det_lines)
        .replace("{{CANDIDATE_RESPONSE}}", raw_response)
    )


def judge(provider: Provider, template: str, case: Case, raw_response: str,
          det: list[DeterministicResult], max_attempts: int = 3) -> tuple[JudgeResult | None, str | None, str]:
    """Returns (result, error, judge_model_name)."""
    prompt = build_judge_prompt(template, case, raw_response, det)
    system = "You are a strict evaluator. Output a single JSON object and nothing else."
    last_error = None
    model_name = provider.model
    for _ in range(max_attempts):
        resp = provider.complete(system, prompt, case_id=case.id, json_only=True)
        model_name = resp.model_name
        try:
            return JudgeResult.model_validate(_parse_json(resp.text)), None, model_name
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = f"{type(e).__name__}: {str(e)[:300]}"
    return None, last_error, model_name


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in judge output")
    return json.loads(text[start : end + 1])


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) or "- none"
