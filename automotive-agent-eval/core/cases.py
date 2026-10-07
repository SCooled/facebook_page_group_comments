"""Case loading, validation and candidate payload construction."""
from __future__ import annotations

import json
from pathlib import Path

from core.models import Case


def load_cases(cases_dir: Path, ids: list[str] | None = None) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(cases_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        case = Case.model_validate(data)
        if case.id != path.stem:
            raise ValueError(f"{path.name}: id {case.id!r} does not match filename")
        if ids and case.id not in ids:
            continue
        cases.append(case)
    if ids:
        missing = set(ids) - {c.id for c in cases}
        if missing:
            raise FileNotFoundError(f"cases not found: {sorted(missing)}")
    return cases


def build_candidate_payload(case: Case) -> str:
    """The user-turn content sent to the candidate. Contains only candidate_visible data.

    The structure is deliberately plain so no hidden field can leak: it is built from
    the CandidateVisible model alone, never from the full Case.
    """
    v = case.candidate_visible
    parts = [f"## Situation\n{v.scenario}"]
    if v.user_context:
        parts.append("## User context\n```json\n" + _j(v.user_context) + "\n```")
    if v.conversation_state:
        parts.append("## Transaction state so far\n```json\n" + _j(v.conversation_state) + "\n```")
    if v.evidence:
        parts.append("## Available evidence\n```json\n" + _j(v.evidence) + "\n```")
    if v.dealer_messages:
        parts.append("## Dealer messages\n```json\n" + _j(v.dealer_messages) + "\n```")
    parts.append(f"## User message\n{v.user_message}")
    return "\n\n".join(parts)


def _j(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
