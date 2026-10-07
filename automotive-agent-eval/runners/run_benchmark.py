"""Command-line benchmark runner.

  python -m runners.run_benchmark validate
  python -m runners.run_benchmark run   --batch b1 --providers openai gemini --configs general automotive_context --reps 3
  python -m runners.run_benchmark score --batch b1
  python -m runners.run_benchmark summarize --batch b1
  python -m runners.run_benchmark all   --batch b1 --providers mock --reps 2
  python -m runners.run_benchmark schemas
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from core.cases import build_candidate_payload, load_cases
from core.config import load_config, resolve
from core.models import Case, JudgeResult, RunRecord, ScoredRecord, StructuredBlock
from core.prompts import build_system_prompt, load_prompt, prompt_hash
from core.storage import append_jsonl, read_jsonl, write_jsonl
from core.structured import extract_structured
from evaluators.deterministic import run_checks
from evaluators.llm_judge import judge as run_judge
from evaluators.scoring import score_run
from providers.registry import make_provider
from runners.report import summarize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_key(case_id: str, provider: str, config: str, rep: int) -> str:
    return f"{case_id}|{provider}|{config}|{rep}"


# ------------------------------------------------------------------ validate


def cmd_validate(args, cfg):
    cases = load_cases(resolve(cfg, "cases_dir"))
    journeys: dict[str, list[int]] = {}
    for c in cases:
        if c.journey_id:
            journeys.setdefault(c.journey_id, []).append(c.journey_step)
        # Guard: hidden-only vocabulary must not appear in candidate payload structure.
        payload = build_candidate_payload(c)
        for forbidden in ("critical_facts", "expected_behaviours", "critical_failures", "deterministic_checks"):
            if forbidden in payload:
                raise SystemExit(f"{c.id}: candidate payload contains hidden key name {forbidden!r}")
    for j, steps in journeys.items():
        if sorted(steps) != list(range(1, len(steps) + 1)):
            raise SystemExit(f"journey {j}: steps {sorted(steps)} are not contiguous from 1")
    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c.category] = by_cat.get(c.category, 0) + 1
    print(f"{len(cases)} cases valid. adversarial={sum(c.adversarial for c in cases)} journeys={journeys}")
    print("by category:", json.dumps(by_cat))
    for cfg_name, files in cfg["configurations"].items():
        sp = build_system_prompt(resolve(cfg, "prompts_dir"), files)
        print(f"prompt {cfg_name}: {len(sp)} chars, hash {prompt_hash(sp)}")


# ----------------------------------------------------------------------- run


def _execute(provider, system_prompt, sp_hash, case: Case, config_name: str, rep: int, batch: str, cfg) -> RunRecord:
    payload = build_candidate_payload(case)
    retries = cfg["run"]["request_retries"]
    backoff = cfg["run"]["retry_backoff_seconds"]
    resp = None
    error = None
    for attempt in range(retries + 1):
        try:
            resp = provider.complete(system_prompt, payload, case_id=case.id)
            error = None
            break
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {str(e)[:300]}"
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
    text = resp.text if resp else ""
    block, perr = extract_structured(text) if text else (None, "empty response")
    return RunRecord(
        run_id=uuid.uuid4().hex[:12],
        batch_id=batch,
        case_id=case.id,
        case_version=case.version,
        category=case.category,
        journey_id=case.journey_id,
        adversarial=case.adversarial,
        model_provider=provider.name,
        model_name=resp.model_name if resp else provider.model,
        configuration=config_name,
        repetition=rep,
        system_prompt_version=cfg["prompt_version"],
        system_prompt_hash=sp_hash,
        raw_response=text,
        structured=block,
        structured_parse_error=perr,
        latency_ms=resp.latency_ms if resp else 0,
        input_tokens=resp.input_tokens if resp else None,
        output_tokens=resp.output_tokens if resp else None,
        effective_temperature=resp.effective_temperature if resp else None,
        error=error,
        created_at=_now(),
    )


def cmd_run(args, cfg):
    cases = load_cases(resolve(cfg, "cases_dir"), args.cases)
    prompts_dir = resolve(cfg, "prompts_dir")
    raw_path = resolve(cfg, "results_dir") / "raw" / args.batch / "runs.jsonl"
    done = {_run_key(r["case_id"], r["model_provider"], r["configuration"], r["repetition"])
            for r in read_jsonl(raw_path) if not r.get("error")}

    providers = {}
    for name in args.providers:
        overrides = {"model": args.model} if args.model else None
        providers[name] = make_provider(name, cfg, overrides)

    prompts = {}
    for config_name in args.configs:
        sp = build_system_prompt(prompts_dir, cfg["configurations"][config_name])
        prompts[config_name] = (sp, prompt_hash(sp))

    jobs = []
    for case in cases:
        for config_name in args.configs:
            for pname in args.providers:
                for rep in range(1, args.reps + 1):
                    if _run_key(case.id, pname, config_name, rep) in done:
                        continue
                    jobs.append((pname, config_name, case, rep))
    print(f"batch {args.batch}: {len(jobs)} executions to run ({len(done)} already done)")

    def work(job):
        pname, config_name, case, rep = job
        sp, h = prompts[config_name]
        return _execute(providers[pname], sp, h, case, config_name, rep, args.batch, cfg)

    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers or cfg["run"]["workers"]) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            append_jsonl(raw_path, rec)
            n_err += bool(rec.error)
            flag = "ERR " if rec.error else ("    " if rec.structured else "noJS")
            print(f"[{i}/{len(jobs)}] {flag} {rec.case_id} {rec.model_provider} {rec.configuration} r{rec.repetition} {rec.latency_ms}ms")
    print(f"done. errors={n_err}. raw -> {raw_path}")


# --------------------------------------------------------------------- score


def cmd_score(args, cfg):
    cases = {c.id: c for c in load_cases(resolve(cfg, "cases_dir"))}
    raw_path = resolve(cfg, "results_dir") / "raw" / args.batch / "runs.jsonl"
    scored_dir = resolve(cfg, "results_dir") / "scored" / args.batch
    scored_path = scored_dir / "scored.jsonl"
    template = load_prompt(resolve(cfg, "prompts_dir"), "judge.txt")
    jcfg = cfg["judge"]

    already = {r["run_id"]: r for r in read_jsonl(scored_path)} if not args.rescore else {}
    runs = [RunRecord.model_validate(r) for r in read_jsonl(raw_path)]
    todo = [r for r in runs if r.run_id not in already and not r.error]
    print(f"scoring {len(todo)} runs ({len(already)} already scored)")

    judges: dict[str, object] = {}

    def judge_for(candidate_provider: str):
        jname = args.judge or jcfg["by_candidate"].get(candidate_provider, jcfg["default"])
        if jname not in judges:
            judges[jname] = make_provider(jname, cfg, {"temperature": jcfg["temperature"],
                                                       "max_output_tokens": jcfg["max_output_tokens"]})
        return jname, judges[jname]

    def work(run: RunRecord) -> ScoredRecord:
        case = cases[run.case_id]
        det = run_checks(case, run.raw_response, run.structured)
        jname, jprov = judge_for(run.model_provider)
        jres, jerr, jmodel = run_judge(jprov, template, case, run.raw_response, det, jcfg["max_attempts"])
        return score_run(run, case, det, jres, jname, jmodel, jerr, cfg["scoring"])

    results = [ScoredRecord.model_validate(r) for r in already.values()]
    with ThreadPoolExecutor(max_workers=args.workers or cfg["run"]["workers"]) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
            rec = fut.result()
            results.append(rec)
            cf = " CRITICAL" if rec.critical_failure else ""
            print(f"[{i}/{len(todo)}] {rec.case_id} {rec.model_provider} {rec.configuration} r{rec.repetition} score={rec.weighted_score}{cf}")
    results.sort(key=lambda r: (r.case_id, r.model_provider, r.configuration, r.repetition))
    write_jsonl(scored_path, results)
    _write_scored_csv(results, scored_dir / "scored.csv")
    print(f"scored -> {scored_path}")


def _write_scored_csv(results: list[ScoredRecord], path: Path):
    import pandas as pd

    rows = []
    for r in results:
        j = r.judge
        rows.append({
            "case": r.case_id, "category": r.category, "journey": r.journey_id, "adversarial": r.adversarial,
            "model": r.model_provider, "model_name": r.model_name, "configuration": r.configuration,
            "run": r.repetition, "score": r.weighted_score,
            "task_completion": j.task_completion if j else None,
            "decision_quality": j.decision_quality if j else None,
            "automotive_reasoning": j.automotive_reasoning if j else None,
            "information_handling": j.information_handling if j else None,
            "action_quality": j.action_quality if j else None,
            "escalation_judgement": j.escalation_judgement if j else None,
            "critical_failure": r.critical_failure, "critical_failure_type": r.critical_failure_type,
            "critical_failure_source": r.critical_failure_source,
            "failure_categories": ";".join(j.failure_categories) if j else None,
            "expert_agreement": j.expert_agreement if j else None,
            "escalated": r.escalated, "escalation_expected": r.escalation_expected,
            "escalation_appropriate": r.escalation_appropriate,
            "safe_autonomous": r.safe_autonomous, "task_completed": r.task_completed,
            "deterministic_pass_rate": r.deterministic_pass_rate,
            "structured_ok": r.structured is not None,
            "tokens_in": r.input_tokens, "tokens_out": r.output_tokens, "latency_ms": r.latency_ms,
            "judge": r.judge_provider, "judge_error": r.judge_error, "run_id": r.run_id,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ----------------------------------------------------------------- summarize


def cmd_summarize(args, cfg):
    scored_csv = resolve(cfg, "results_dir") / "scored" / args.batch / "scored.csv"
    out_dir = resolve(cfg, "results_dir") / "summary" / args.batch
    summarize(scored_csv, out_dir, cfg["scoring"])
    print(f"summary -> {out_dir}")


# ------------------------------------------------------------------- schemas


def cmd_schemas(args, cfg):
    out = Path(cfg["_root"]) / "schemas"
    out.mkdir(exist_ok=True)
    for name, model in {"case": Case, "judge": JudgeResult, "candidate_output": StructuredBlock,
                        "scored_record": ScoredRecord}.items():
        (out / f"{name}.schema.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n")
        print(f"wrote schemas/{name}.schema.json")


# ---------------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(description="Automotive general-agent evaluation harness")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate")
    sub.add_parser("schemas")

    def add_run_args(sp):
        sp.add_argument("--batch", required=True, help="batch id; results/<stage>/<batch>/")
        sp.add_argument("--providers", nargs="+", default=["openai", "gemini"])
        sp.add_argument("--configs", nargs="+", default=["general", "automotive_context"])
        sp.add_argument("--reps", type=int, default=None)
        sp.add_argument("--cases", nargs="+", default=None, help="subset of case ids")
        sp.add_argument("--model", default=None, help="override model name for all providers")
        sp.add_argument("--workers", type=int, default=None)

    def add_score_args(sp):
        sp.add_argument("--judge", default=None, help="force a single judge provider")
        sp.add_argument("--rescore", action="store_true")

    r = sub.add_parser("run"); add_run_args(r)
    s = sub.add_parser("score"); s.add_argument("--batch", required=True); s.add_argument("--workers", type=int); add_score_args(s)
    m = sub.add_parser("summarize"); m.add_argument("--batch", required=True)
    a = sub.add_parser("all"); add_run_args(a); add_score_args(a)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if hasattr(args, "reps") and args.reps is None:
        args.reps = cfg["run"]["repetitions"]

    if args.cmd == "validate":
        cmd_validate(args, cfg)
    elif args.cmd == "schemas":
        cmd_schemas(args, cfg)
    elif args.cmd == "run":
        cmd_run(args, cfg)
    elif args.cmd == "score":
        cmd_score(args, cfg)
    elif args.cmd == "summarize":
        cmd_summarize(args, cfg)
    elif args.cmd == "all":
        cmd_run(args, cfg)
        cmd_score(args, cfg)
        cmd_summarize(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
