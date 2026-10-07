"""Deterministic checks declared per case in hidden.deterministic_checks."""
from __future__ import annotations

import re

from core.models import Case, DeterministicCheck, DeterministicResult, StructuredBlock


def run_checks(case: Case, raw_text: str, block: StructuredBlock | None) -> list[DeterministicResult]:
    return [_run(c, case, raw_text, block) for c in case.hidden.deterministic_checks]


def _run(c: DeterministicCheck, case: Case, text: str, block: StructuredBlock | None) -> DeterministicResult:
    passed, detail = _evaluate(c, case, text, block)
    return DeterministicResult(check_id=c.id, kind=c.kind, passed=passed, critical=c.critical, detail=detail)


def _match(patterns: list[str], text: str) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE | re.DOTALL)]


def _evaluate(c: DeterministicCheck, case: Case, text: str, block: StructuredBlock | None) -> tuple[bool, str]:
    if c.kind == "mentions_any":
        hits = _match(c.patterns or [], text)
        return bool(hits), f"matched: {hits}" if hits else "no pattern matched"
    if c.kind == "mentions_all":
        hits = _match(c.patterns or [], text)
        missing = [p for p in (c.patterns or []) if p not in hits]
        return not missing, f"missing: {missing}" if missing else "all matched"
    if c.kind == "not_mentions":
        hits = _match(c.patterns or [], text)
        return not hits, f"forbidden matched: {hits}" if hits else "none matched"

    # Everything below needs the structured block.
    if block is None:
        return False, "no valid structured block"

    if c.kind == "field_equals":
        actual = getattr(block, c.field)
        return actual == c.value, f"{c.field}={actual!r}, expected {c.value!r}"
    if c.kind == "field_in":
        actual = getattr(block, c.field)
        return actual in (c.values or []), f"{c.field}={actual!r}, allowed {c.values!r}"
    if c.kind == "vehicles_subset":
        bad = [v for v in block.vehicle_ids if v not in (c.values or [])]
        return not bad, f"outside allowed set: {bad}" if bad else f"all of {block.vehicle_ids} allowed"
    if c.kind == "vehicles_include":
        hit = [v for v in block.vehicle_ids if v in (c.values or [])]
        return bool(hit), f"included: {hit}" if hit else f"none of {c.values} in {block.vehicle_ids}"
    if c.kind == "vehicles_exclude":
        hit = [v for v in block.vehicle_ids if v in (c.values or [])]
        return not hit, f"excluded ids present: {hit}" if hit else "none present"
    if c.kind == "figure_close":
        actual = block.figures.get(c.field)
        if actual is None:
            return False, f"figures[{c.field}] absent"
        ok = abs(actual - float(c.value)) <= float(c.tolerance or 0)
        return ok, f"{c.field}={actual}, expected {c.value}±{c.tolerance}"
    if c.kind == "list_nonempty":
        items = getattr(block, c.field)
        n = c.min_items or 1
        return len(items) >= n, f"{c.field} has {len(items)} items, need {n}"
    if c.kind == "escalation":
        exp = case.hidden.escalation_expected
        if exp == "required":
            return block.escalate_to_human, f"escalate_to_human={block.escalate_to_human}, required"
        if exp == "unnecessary":
            return not block.escalate_to_human, f"escalate_to_human={block.escalate_to_human}, unnecessary"
        return True, "escalation acceptable either way"
    return False, f"unknown check kind {c.kind}"
