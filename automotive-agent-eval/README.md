# Automotive General-Agent Evaluation Harness

Tests whether a strong general-purpose model can safely handle most of a UK used-car purchase
journey, how much a thin layer of automotive context helps, and where a human remains necessary.
The full brief is in `BRIEF.md`; the economic reframing (cost, latency, reliability, throughput and
the specialist advantage vector) is in `BRIEF_ECONOMICS.md`; the cognitive primitive map is in `PRIMITIVES.md`.

## What it does

```
cases/*.json  ──►  runner  ──►  provider adapters (openai | gemini | instinct | human | mock)
                                   │
                              results/raw/<batch>/runs.jsonl
                                   │
                    deterministic checks + blind LLM judge
                                   │
                              results/scored/<batch>/scored.{jsonl,csv}
                                   │
                              results/summary/<batch>/{summary.json, report.md, *.csv}
```

Each case has a `candidate_visible` half (all the model sees) and a `hidden` half (critical facts,
expected behaviours, critical-failure conditions, escalation ground truth, deterministic checks,
expert notes). The payload builder reads only `candidate_visible`, so leakage is structurally impossible.

Candidates answer in prose and append a JSON block (`prompts/output_format.txt`) with recommendation,
`escalate_to_human`, ranked `vehicle_ids`, questions and figures. Deterministic checks run on that block
and on the prose; the judge scores six 0–4 dimensions, flags critical failures and tags failure categories.

## Setup

```
cd automotive-agent-eval
pip install -e .            # or: pip install pydantic pandas pyyaml openai google-genai tabulate
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
python -m runners.run_benchmark validate
python -m pytest -q         # offline tests with the mock provider
```

Model names, temperature, token limits, judge routing, repetitions, weights and the safe-autonomous
threshold are all in `config.yaml`.

## Run

```
# smoke test, no API keys
python -m runners.run_benchmark all --batch smoke --providers mock --reps 1

# the 24+6 cases x 2 configurations x 2 providers x 3 reps
python -m runners.run_benchmark run       --batch b1 --providers openai gemini --configs general automotive_context --reps 3
python -m runners.run_benchmark score     --batch b1            # judge routed per config.yaml (openai->gemini, gemini->openai)
python -m runners.run_benchmark summarize --batch b1

# subsets, resumable: already-completed executions are skipped
python -m runners.run_benchmark run --batch b1 --providers openai --cases S16 S23 S24 --reps 1
```

`results/` is git-ignored. Every run record carries provider, model name, configuration, repetition,
prompt version and hash, case version, raw response, parsed block, tokens, latency and effective temperature.

## Implementations and conditions

| | How |
|---|---|
| A general | `--configs general`: `prompts/general_agent.txt` + output format |
| B general + automotive context | `--configs automotive_context`: adds `prompts/automotive_context.txt` (principles + checklist, deliberately thin) |
| C general + specialist primitives | `--configs delegated`: B plus `core/primitives.py` engines (hard-constraint filter, duplicate detector, heuristic ranker, comparable valuation) pre-process the evidence; their cost and latency are charged to the run |
| Human expert | `--providers human`: reads `human_responses/<case>.md`, scored by the same blind judge; cost and time per task from `config.yaml` |

Instinct is a placeholder adapter: set `providers.instinct.base_url` (OpenAI-compatible assumed) and `INSTINCT_API_KEY`.

## Metrics (results/summary/<batch>/)

- **safe_autonomous_coverage** (primary): no critical failure AND weighted score ≥ 70 AND task_completion ≥ 3 AND decision_quality ≥ 3. Thresholds in `config.yaml`.
- autonomous_completion_rate: task_completion ≥ 3.
- critical_failure_rate: judge-flagged or a `critical: true` deterministic check failed.
- appropriate_escalation_rate: structured `escalate_to_human` versus the case's `escalation_expected` (`required` / `unnecessary` / `acceptable`).
- expert_agreement_rate: judge's estimate until human calibration replaces it.
- Breakdowns by provider, configuration, category (journey stage), case, run, adversarial; variance across repetitions; failure-category and critical-failure-type clustering; `delta_context_vs_general`.
- human_intervention_rate is a manual metric from the calibration sheet.

### Economic vector (results/summary/<batch>/economics_*.csv, esa_*.csv)

Every run records tokens, model cost (from the pricing table in `config.yaml`), primitive cost and
latency, objects in scope and validity. Per implementation, per primary cognitive primitive, per
expected class and per case the summary reports Q quality, C cost per task / per safe completion /
per object, L p50 and p95, R reliability, T objects per second and tasks per hour, and utility U
under each weight profile. `esa_vs_baseline.csv` and `esa_by_primitive.csv` give the economic
specialist advantage of B and C over A. Judge cost is recorded separately as evaluation overhead.

## Judge calibration

```
python -m runners.calibrate export  --batch b1 --fraction 0.15   # blind sheet + reading bundle
# human fills human_* columns in results/summary/b1/calibration_sheet.csv
python -m runners.calibrate compare --batch b1                   # disagreement by dimension and critical failures
```

## Cases

24 core scenarios `S01`–`S24` and 6 adversarial variants `A01`–`A06`, each tagged with cognitive
primitives (`primitives`, `primary_primitive`, `expected_class`; see `PRIMITIVES.md`). `S01, S05, S11, S19, S20, S16, S23, S24`
form journey `J1` (one buyer, one Kodiaq, intent through to deposit decision) with seeded state at each step.
Authoring rules: `cases/AUTHORING.md`. Listing data is synthetic; the human expert should review prices and
hidden answer keys before results are relied on.

## Layout

```
cases/        S01..S24.json, A01..A06.json, AUTHORING.md
prompts/      general_agent.txt, automotive_context.txt, delegated.txt, output_format.txt, judge.txt
providers/    base.py, openai_adapter.py, gemini_adapter.py, instinct_adapter.py, human_adapter.py, mock_adapter.py, registry.py
evaluators/   deterministic.py, llm_judge.py, scoring.py, economics.py
runners/      run_benchmark.py (CLI), report.py, calibrate.py
core/         models.py (schemas), cases.py, primitives.py, prompts.py, structured.py, storage.py, config.py
schemas/      generated JSON schemas (python -m runners.run_benchmark schemas)
human_responses/  Condition C answers
results/      raw/ scored/ summary/ per batch (git-ignored)
```

## Deliberate choices

- No tools in the MVP: all evidence is in the case payload, so providers are compared on equal footing.
- Seeded journey state, not live chaining, so repetitions stay comparable. Live chaining can be added as a mode.
- Judge never sees provider or configuration. Cross-provider judging by default.
- Temperature is requested but models that reject it are retried without; the effective value is recorded.
