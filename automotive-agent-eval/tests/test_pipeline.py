"""Offline tests: schema, payload isolation, structured parsing, deterministic checks, scoring, CLI with mock provider."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cases import build_candidate_payload, load_cases  # noqa: E402
from core.models import Case, DeterministicCheck, JudgeResult, StructuredBlock  # noqa: E402
from core.structured import extract_structured  # noqa: E402
from evaluators.deterministic import run_checks  # noqa: E402
from evaluators.scoring import weighted_score  # noqa: E402

WEIGHTS = {"task_completion": 0.15, "decision_quality": 0.25, "automotive_reasoning": 0.20,
           "information_handling": 0.15, "action_quality": 0.15, "escalation_judgement": 0.10}


def test_all_cases_load_and_hide_evaluator_data():
    cases = load_cases(ROOT / "cases")
    assert cases, "no cases"
    for c in cases:
        payload = build_candidate_payload(c)
        for fact in c.hidden.critical_facts:
            assert fact not in payload, f"{c.id}: critical fact leaked verbatim"
        for key in ("critical_facts", "expected_behaviours", "critical_failures", "deterministic_checks", "expert_notes"):
            assert key not in payload, f"{c.id}: hidden key {key} in payload"
        assert c.candidate_visible.user_message.strip()


def test_structured_extraction_takes_last_valid_block():
    text = 'prose\n```json\n{"foo": 1}\n```\nmore\n```json\n{"recommendation": "negotiate", "escalate_to_human": false}\n```'
    block, err = extract_structured(text)
    assert err is None and block.recommendation == "negotiate"


def test_structured_extraction_reports_error():
    block, err = extract_structured("no json here")
    assert block is None and err


def _case(checks, escalation="unnecessary"):
    return Case(id="T1", category="pricing", title="t",
                candidate_visible={"scenario": "s", "user_message": "u"},
                hidden={"critical_facts": [], "expected_behaviours": [], "critical_failures": [],
                        "escalation_expected": escalation,
                        "deterministic_checks": [DeterministicCheck(**c) for c in checks]})


def test_deterministic_checks():
    block = StructuredBlock(recommendation="negotiate", escalate_to_human=False, vehicle_ids=["L07", "L02"],
                            figures={"opening_offer_gbp": 20900})
    case = _case([
        {"id": "a", "kind": "mentions_any", "description": "", "patterns": ["SE ?L"]},
        {"id": "b", "kind": "not_mentions", "description": "", "patterns": ["bargain"]},
        {"id": "c", "kind": "field_in", "description": "", "field": "recommendation", "values": ["negotiate"]},
        {"id": "d", "kind": "vehicles_subset", "description": "", "values": ["L07", "L02", "L15"]},
        {"id": "e", "kind": "vehicles_exclude", "description": "", "values": ["L02"], "critical": True,
         "critical_failure_type": "ignored_decisive_evidence"},
        {"id": "f", "kind": "figure_close", "description": "", "field": "opening_offer_gbp", "value": 21000, "tolerance": 200},
        {"id": "g", "kind": "escalation", "description": ""},
    ])
    res = {r.check_id: r for r in run_checks(case, "The SE L spec justifies it.", block)}
    assert res["a"].passed and res["b"].passed and res["c"].passed and res["d"].passed
    assert not res["e"].passed and res["e"].critical
    assert res["f"].passed and res["g"].passed


def test_weighted_score_formula():
    j = JudgeResult(task_completion=4, decision_quality=3, automotive_reasoning=3, information_handling=4,
                    action_quality=3, escalation_judgement=4, critical_failure=False, expert_agreement=True,
                    reasoning_summary="x")
    assert weighted_score(j, WEIGHTS) == 85.0


def test_judge_requires_type_when_critical():
    with pytest.raises(Exception):
        JudgeResult(task_completion=1, decision_quality=1, automotive_reasoning=1, information_handling=1,
                    action_quality=1, escalation_judgement=1, critical_failure=True, expert_agreement=False,
                    reasoning_summary="x")


def test_cli_end_to_end_with_mock(tmp_path):
    batch = "pytest_mock"
    for stage in ("raw", "scored", "summary"):
        shutil.rmtree(ROOT / "results" / stage / batch, ignore_errors=True)
    cmd = [sys.executable, "-m", "runners.run_benchmark", "all", "--batch", batch, "--providers", "mock",
           "--configs", "general", "automotive_context", "--reps", "2", "--cases", "S16"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    summary = json.loads((ROOT / "results" / "summary" / batch / "summary.json").read_text())
    assert summary["rows"] == 4
    assert set(summary["headline_by_configuration"]) == {"general", "automotive_context"}
    scored = (ROOT / "results" / "scored" / batch / "scored.csv").read_text()
    assert "weighted" not in scored and "score" in scored
    for stage in ("raw", "scored", "summary"):
        shutil.rmtree(ROOT / "results" / stage / batch, ignore_errors=True)
