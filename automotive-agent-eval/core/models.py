"""Pydantic models for cases, candidate output, judge output and run records."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal[
    "intent",
    "search",
    "substitution",
    "evaluation",
    "pricing",
    "dealer_interaction",
    "offer_comparison",
    "negotiation",
    "escalation",
]

EscalationExpectation = Literal["required", "acceptable", "unnecessary"]

FailureCategory = Literal[
    "requirements_interpretation",
    "search",
    "substitution",
    "vehicle_specification",
    "vehicle_history",
    "valuation",
    "comparator_selection",
    "risk_assessment",
    "finance",
    "total_transaction_economics",
    "dealer_behaviour",
    "negotiation",
    "uncertainty_handling",
    "escalation",
    "hallucination_unsupported_fact",
    "state_management",
]

CriticalFailureType = Literal[
    "invented_vehicle_fact",
    "missed_safety_issue",
    "missed_provenance_issue",
    "incorrect_financial_reasoning",
    "proceeds_despite_missing_information",
    "missed_listing_discrepancy",
    "unsupported_mechanical_conclusion",
    "unsupported_market_value",
    "asking_price_treated_as_fair_value",
    "exceeded_authority",
    "failed_to_escalate",
    "misinterpreted_total_cost",
    "ignored_decisive_evidence",
]


# --------------------------------------------------------------------------- cases


class DeterministicCheck(BaseModel):
    """A machine-checkable assertion about the candidate response.

    kinds:
      mentions_any      - at least one regex in `patterns` matches the full response text
      mentions_all      - every regex in `patterns` matches
      not_mentions      - no regex in `patterns` matches
      field_equals      - structured block field `field` == `value`
      field_in          - structured block field `field` is one of `values`
      vehicles_subset   - every id in structured `vehicle_ids` is within `values`
      vehicles_include  - structured `vehicle_ids` contains at least one of `values`
      vehicles_exclude  - structured `vehicle_ids` contains none of `values`
      figure_close      - structured `figures[field]` within `tolerance` of `value`
      escalation        - structured `escalate_to_human` consistent with hidden escalation_expected
      list_nonempty     - structured list field `field` has at least `min_items` entries
    """

    id: str
    kind: Literal[
        "mentions_any",
        "mentions_all",
        "not_mentions",
        "field_equals",
        "field_in",
        "vehicles_subset",
        "vehicles_include",
        "vehicles_exclude",
        "figure_close",
        "escalation",
        "list_nonempty",
    ]
    description: str
    patterns: list[str] | None = None
    field: str | None = None
    value: Any | None = None
    values: list[Any] | None = None
    tolerance: float | None = None
    min_items: int | None = None
    # If true, failing this check marks the run as a critical failure.
    critical: bool = False
    critical_failure_type: CriticalFailureType | None = None


class CandidateVisible(BaseModel):
    """Everything the candidate model is allowed to see."""

    scenario: str
    user_context: dict[str, Any] = Field(default_factory=dict)
    conversation_state: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    dealer_messages: list[dict[str, Any]] = Field(default_factory=list)
    user_message: str


class Hidden(BaseModel):
    """Evaluator-only data. Never sent to the candidate."""

    critical_facts: list[str]
    expected_behaviours: list[str]
    critical_failures: list[str]
    escalation_expected: EscalationExpectation
    deterministic_checks: list[DeterministicCheck] = Field(default_factory=list)
    expert_notes: str | None = None
    failure_categories_at_risk: list[FailureCategory] = Field(default_factory=list)


class Case(BaseModel):
    id: str
    version: str = "v1"
    category: Category
    title: str
    adversarial: bool = False
    variant_of: str | None = None
    journey_id: str | None = None
    journey_step: int | None = None
    candidate_visible: CandidateVisible
    hidden: Hidden

    @model_validator(mode="after")
    def _journey_fields_consistent(self) -> "Case":
        if (self.journey_id is None) != (self.journey_step is None):
            raise ValueError("journey_id and journey_step must be set together")
        return self


# ------------------------------------------------------------- candidate output


class StructuredBlock(BaseModel):
    """The JSON block the candidate must append after its prose answer."""

    recommendation: Literal[
        "clarify_requirements",
        "shortlist",
        "gather_information",
        "proceed",
        "proceed_with_conditions",
        "negotiate",
        "reject",
        "escalate",
    ]
    escalate_to_human: bool
    escalation_reason: str | None = None
    vehicle_ids: list[str] = Field(default_factory=list)
    questions_for_dealer: list[str] = Field(default_factory=list)
    questions_for_user: list[str] = Field(default_factory=list)
    figures: dict[str, float] = Field(default_factory=dict)
    unresolved_risks: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ judge output


class JudgeResult(BaseModel):
    task_completion: int = Field(ge=0, le=4)
    decision_quality: int = Field(ge=0, le=4)
    automotive_reasoning: int = Field(ge=0, le=4)
    information_handling: int = Field(ge=0, le=4)
    action_quality: int = Field(ge=0, le=4)
    escalation_judgement: int = Field(ge=0, le=4)
    critical_failure: bool
    critical_failure_type: CriticalFailureType | None = None
    critical_failure_reason: str | None = None
    failure_categories: list[FailureCategory] = Field(default_factory=list)
    expert_agreement: bool
    reasoning_summary: str

    @model_validator(mode="after")
    def _cf_fields(self) -> "JudgeResult":
        if self.critical_failure and not self.critical_failure_type:
            raise ValueError("critical_failure_type required when critical_failure is true")
        return self


# ------------------------------------------------------------------- run records


class ProviderResponse(BaseModel):
    text: str
    model_name: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    effective_temperature: float | None = None
    error: str | None = None


class RunRecord(BaseModel):
    """One candidate execution. Stored in results/raw/<batch>/runs.jsonl."""

    run_id: str
    batch_id: str
    case_id: str
    case_version: str
    category: str
    journey_id: str | None
    adversarial: bool
    model_provider: str
    model_name: str
    configuration: str
    repetition: int
    system_prompt_version: str
    system_prompt_hash: str
    raw_response: str
    structured: StructuredBlock | None
    structured_parse_error: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    effective_temperature: float | None
    error: str | None
    created_at: str


class DeterministicResult(BaseModel):
    check_id: str
    kind: str
    passed: bool
    critical: bool
    detail: str


class ScoredRecord(RunRecord):
    """RunRecord plus evaluator outputs. Stored in results/scored/<batch>/scored.jsonl."""

    deterministic: list[DeterministicResult] = Field(default_factory=list)
    deterministic_pass_rate: float | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    judge: JudgeResult | None = None
    judge_error: str | None = None
    weighted_score: float | None = None
    critical_failure: bool = False
    critical_failure_source: Literal["judge", "deterministic", "both"] | None = None
    critical_failure_type: str | None = None
    escalated: bool | None = None
    escalation_expected: str
    escalation_appropriate: bool | None = None
    safe_autonomous: bool | None = None
    task_completed: bool | None = None
