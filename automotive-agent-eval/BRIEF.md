# Automotive General-Agent Evaluation Harness — build brief

## 1. Objective

Build an automated evaluation harness to test whether a strong general-purpose AI agent is capable of handling the majority of an automotive purchase journey before handing off to a human.

Core hypothesis: a general-purpose agent may be sufficiently capable for most of the automotive transaction journey, provided it has access to the relevant information and tools, with human intervention reserved for higher-risk, ambiguous, or commercially sensitive stages. The purpose is to test that empirically rather than assume automotive requires a deeply specialised agent architecture.

Compare multiple general-purpose models, initially OpenAI, Google Gemini, Instinct, and others later. Determine:

1. How much of the automotive journey a general agent can complete correctly.
2. Where automotive-specific knowledge or tooling materially improves performance.
3. Where human intervention remains necessary.
4. Whether the remaining specialist capability is broad and fundamental, or concentrated in a small number of nodes such as valuation, negotiation, specification interpretation, or risk assessment.

## 2. Strategic context

Architecture under investigation: general-purpose agent → automotive-specific context / tools / workflow → automotive and proprietary data. The question is how substantial the middle layer needs to be.

Generic capabilities: communications, scheduling, payments, identity handoff, basic workflow orchestration.
Automotive-specific capabilities (initial specialist cluster): decomposing a vehicle search, vehicle substitution logic, fair-price assessment, listing-history interpretation, risk checks, offer evaluation, negotiation strategy.
Potential proprietary-data layer: historical listing changes, dealer-response behaviour, actual sale prices, negotiation outcomes, conversion outcomes, transaction behaviour.

Risk: the layers are less separable than they look. Communication and scheduling depend on automotive transaction state (availability, test-drive status, dealer opening patterns, preparation, finance availability, part exchange, deposits, licence checks, KYC, insurance, stock location, negotiation state). Search and evaluation need automotive ontology (generations, derivatives, trims, options, engines, transmissions, drivetrain, model year vs registration year, mileage, service history, warranty, tax, emissions-zone eligibility, recalls, known faults, ownership history, regional pricing, depreciation, running costs).

## 3. Hypotheses

- **H1 General-agent sufficiency.** A strong general-purpose agent can correctly execute most of the journey before human handoff. Core measurement: **safe autonomous coverage** = percentage of transaction stages completed correctly without material human correction or critical failure.
- **H2 Thin automotive context may be sufficient.** General agent + small automotive context (terminology, workflow, evaluation checklist, risk checks, comparison methodology, uncertainty/escalation rules) performs approximately as well as a specialist human.
- **H3 Specialist automotive reasoning remains necessary.** If both A and B perform materially below an experienced human operator, there is evidence for a dedicated automotive reasoning layer. Identify exactly where the gap is.

## 4. Experimental conditions

| Condition | Description |
|---|---|
| A General agent | Strong general model, standard system instructions, general tools |
| B General + automotive context | Same model plus automotive glossary, workflow checklist, evaluation framework, domain guidance |
| C Human expert | Experienced automotive operator as reference benchmark |

Keep everything else constant across AI conditions: case information, prompt structure, tools, context, output constraints, temperature, maximum response length, interaction history. Candidates never see the scoring criteria or hidden answer key.

## 5. Model comparison

Provider adapters behind a generic `run_model(provider, system_prompt, case)` interface. Easy to add providers. Instinct gets a placeholder adapter until its API is known.

## 6–7. Test set: 24 core scenarios

Not trivia; each tests a sensible transactional decision.

**Intent and needs decomposition**
- S1 Basic family-car requirement: "I need a family car for under £25k." Elicitation, missing constraints, no premature recommendation. Hidden difficulty: budget alone does not define a sensible search.
- S2 Soft preference decomposition: "I want something sporty but practical." Translate into criteria (performance, body style, rear-seat usability, boot, comfort, running cost, premium positioning).
- S3 Conflicting requirements: premium, large SUV, very economical, low budget, low running costs. Identify incompatibility, prioritise, negotiate requirements.
- S4 User does not know the segment: reason from usage to hatchback / estate / saloon / crossover / SUV / MPV.

**Search**
- S5 Vehicle shortlist from a substantial listing set: five suitable cars; hard constraints, soft preferences, ranking, utility not keywords.
- S6 Duplicate listings: same vehicle on different platforms with inconsistent metadata; deduplication, entity matching.
- S7 Better option outside preferred brand: distinguish hard constraints from weak preferences.
- S8 Correct model, wrong derivative: engine, trim, drivetrain, equipment or body style changes suitability.

**Substitution**
- S9 Preferred vehicle unavailable: substitutes that preserve underlying utility.
- S10 Superficially similar cheaper alternative, materially weaker where the user cares; attribute weighting.

**Vehicle evaluation**
- S11 Age / mileage / specification trade-off (A newer, higher mileage, lower trim vs B older, lower mileage, better spec).
- S12 Unusually low mileage on an older car: positive, neutral or concerning; investigate.
- S13 Specification inconsistency: advertised trim conflicts with equipment shown.
- S14 Missing service history: information-gap detection, ask for evidence.
- S15 Multiple previous owners: contextual reasoning, not automatically disqualifying.

**Pricing**
- S16 £1,500 above comparables but materially better spec or provenance; avoid naive median logic.
- S17 Cheap but stale listing: interpret listing duration; low price is not sufficient evidence of value.
- S18 Repeated price reductions over six weeks: seller motivation, negotiation reasoning.

**Dealer interaction**
- S19 Missing listing details: what to ask the dealer; prioritised, not generic.
- S20 Evasive dealer response: detect unresolved uncertainty, quality follow-up, escalate or reject where appropriate.

**Offer comparison**
- S21 Price versus bundled value (warranty, service, preparation).
- S22 Finance-supported price vs cash price: effective transaction economics.

**Negotiation**
- S23 Preparing an offer: opening, target, reservation, walk-away, evidence-based; no arbitrary discount.

**Escalation**
- S24 Attractive vehicle with unresolved uncertainties: proceed or hand off. The agent must know when to stop.

## 8. Adversarial variants (at least six)

1. Misleading listing (advertised trim higher than spec supports).
2. Dealer anchoring ("already £2,000 below market").
3. Scarcity pressure ("another customer is coming this afternoon").
4. Bad comparator (cheapest comparison differs materially).
5. User bias (user strongly prefers the objectively weaker option).
6. Unsupported history claim ("full service history", no records).

## 9. Case structure

Structured JSON. Candidate receives only: scenario, user context, conversation state, listing information, dealer messages, realistically available evidence, user request. Candidate must not receive: critical facts, expected behaviours, expert answer, scoring rubric, critical-failure conditions.

## 10. Linked journeys

Several cases form multi-step journeys with the same user and vehicle (requirements → search → shortlist → comparison → dealer query → dealer response → price assessment → offer → escalation). Support both independent and chained scenarios.

## 11–12. Scoring rubric and weights

Six dimensions scored 0–4: task completion (15%), decision quality (25%), automotive reasoning (20%), information handling (15%), action quality (15%), escalation judgement (10%). Weighted score = Σ(dim × weight) / 4 × 100. Level descriptors are in `prompts/judge.txt`.

## 13. Critical failure flags

Averages must not hide serious errors. Critical failure when the agent: invents material vehicle facts; misses a clear safety issue; misses an important provenance issue; gives materially incorrect financial reasoning; recommends proceeding despite decision-critical missing information; fails to identify a major listing discrepancy; presents unsupported mechanical conclusions as fact; gives unjustified precise market valuations; treats asking price as fair value without reasoning; negotiates or commits beyond authority; fails to escalate where a competent operator clearly should; misinterprets finance or total cost; ignores evidence that significantly changes the recommendation. Store `critical_failure`, `critical_failure_type`, `critical_failure_reason`.

## 14. Primary metrics

Safe autonomous coverage (primary); autonomous completion rate; critical failure rate; human intervention rate; appropriate escalation rate (both failure to escalate and unnecessary escalation); expert agreement rate; performance by journey stage; variance across repetitions.

## 15. Indicative thresholds

| Safe autonomous coverage | Interpretation |
|---|---|
| > 85% | Strong evidence a general agent is sufficient for most of the journey |
| 70–85% | General agent likely viable with targeted automotive context/tools |
| 50–70% | Significant specialist automotive capability probably required |
| < 50% | General agent alone is not credible |

Critical failure rate above ~5% is a major deployment concern regardless of mean score. Acceptable thresholds may vary by task type.

## 16–17. Execution and repetitions

Use provider APIs, never manual copy-paste. cases → runner → adapters → raw outputs → deterministic checks + LLM judge → scores → CSV/JSON/report. Run each scenario at least three times (3–5). Initial: 24 × 3 models × 2 configurations × 3 runs = 432 executions.

## 18–20. Automated evaluation, evaluator bias, human calibration

A. Deterministic checks where possible (discrepancy identified, missing evidence mentioned, finance totals correct, cheapest vehicle identified, required question asked, escalated, unsupported precise value stated, vehicle outside hard budget recommended).
B. LLM judge receives scenario, candidate-visible evidence, hidden facts, expected behaviours, critical-failure conditions, candidate response and rubric; returns strict JSON validated against a schema; retry or mark invalid.
Avoid self-judging: use a different provider to judge, or two judges averaged, or one strong judge plus periodic human validation. Manually score 10–20% of outputs; measure absolute, per-category and critical-failure disagreement; refine judge instructions.

## 21–22. Output

Store every execution with reproducibility metadata (run id, case id, category, provider, model, configuration, repetition, prompt version, case version, raw response, latency, tokens, six scores, weighted score, critical failure). Produce CSV and JSON; aggregate by provider, configuration, scenario, category, run, failure type.

## 23. Key comparisons

General vs general + automotive context (Δ score, Δ safe autonomous coverage, Δ critical failures, Δ escalation accuracy). AI vs human expert: is the gap broad or concentrated in a few specialist tasks?

## 24. Failure clustering

Taxonomy: requirements interpretation, search, substitution, vehicle specification, vehicle history, valuation, comparator selection, risk assessment, finance, total transaction economics, dealer behaviour, negotiation, uncertainty handling, escalation, hallucination / unsupported fact, state-management failure. Report frequency and severity per class.

## 25. Automotive context condition

Deliberately lightweight: operating principles (hard constraints vs preferences, verify derivative, do not trust listing claims, advertised price ≠ fair value, like-for-like comparison, identify missing evidence, fact vs inference, total economics, warranty/preparation/finance, evidence-based negotiation, escalate unresolved high-risk issues) and an evaluation checklist (fit, specification, age, mileage, history, condition evidence, ownership, warranty, price, comparables, listing history, dealer information, finance, part exchange, unknowns, risks, next action). Not a hidden specialist agent.

## 26–27. What matters, what to avoid

Strongest result: general agent good but imperfect; + lightweight context approximately human-level on most scenarios; human only materially superior on a few specialist tasks → thin architecture. If lightweight context does not close the gap → invest in a substantial automotive task engine.

Do not optimise the test to make specialist software look necessary. Avoid weak prompts, withholding information a real system could access, artificially limiting tools, trivia, rewarding style or length, assuming disagreement with the human means failure, only clean listings, letting the judge see the provider.

## 28–30. Structure, MVP, purpose

MVP: JSON case format, initial cases, OpenAI and Gemini adapters, placeholder Instinct adapter, general prompt, automotive-context prompt, LLM judge, weighted scoring, critical-failure detection, CSV output, CLI runner. Then expand to all 24 scenarios and adversarial variants.

The benchmark informs an architectural and investment decision: how much of an automotive purchase transaction genuinely requires automotive-specific intelligence, where human intervention is required, which specialist capabilities warrant proprietary implementation, and where proprietary transaction data would measurably help. The output should support a statement like: "A general agent completed 81% of the tested journey safely. Lightweight automotive context increased this to 91%. Residual failures were concentrated in valuation, variant interpretation and negotiation."
