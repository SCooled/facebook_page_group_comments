"""Human calibration of the LLM judge (brief section 20).

  python -m runners.calibrate export  --batch b1 --fraction 0.15   -> results/summary/b1/calibration_sheet.csv
  (a human fills in the human_* columns)
  python -m runners.calibrate compare --batch b1                   -> results/summary/b1/calibration_report.md

The export is blind: provider and configuration are withheld from the sheet (kept in a
separate key file) so the human scores without knowing which model produced the answer.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from core.cases import build_candidate_payload, load_cases
from core.config import load_config, resolve
from core.storage import read_jsonl
from evaluators.scoring import DIMENSIONS, weighted_score
from core.models import JudgeResult

HUMAN_COLS = [f"human_{d}" for d in DIMENSIONS] + ["human_critical_failure", "human_expert_agreement",
                                                   "human_needs_intervention", "human_failure_categories", "human_notes"]


def cmd_export(args, cfg):
    scored = list(read_jsonl(resolve(cfg, "results_dir") / "scored" / args.batch / "scored.jsonl"))
    scored = [r for r in scored if r.get("judge")]
    rng = random.Random(args.seed)
    # Stratify by case so every scenario is represented.
    by_case: dict[str, list] = {}
    for r in scored:
        by_case.setdefault(r["case_id"], []).append(r)
    sample = []
    for case_id, rows in sorted(by_case.items()):
        k = max(1, round(len(rows) * args.fraction))
        sample += rng.sample(rows, k)
    rng.shuffle(sample)
    cases = {c.id: c for c in load_cases(resolve(cfg, "cases_dir"))}

    out_dir = resolve(cfg, "results_dir") / "summary" / args.batch
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_rows, key_rows, bundle = [], [], []
    for i, r in enumerate(sample, 1):
        sid = f"CAL{i:03d}"
        sheet_rows.append({"sample_id": sid, "case_id": r["case_id"], **{c: "" for c in HUMAN_COLS}})
        key_rows.append({"sample_id": sid, "run_id": r["run_id"], "model_provider": r["model_provider"],
                         "configuration": r["configuration"], "repetition": r["repetition"]})
        c = cases[r["case_id"]]
        bundle.append(f"# {sid} ({r['case_id']})\n\n## Candidate input\n\n{build_candidate_payload(c)}\n\n"
                      f"## Hidden evaluator data\n\n```json\n{json.dumps(c.hidden.model_dump(exclude={'deterministic_checks'}), indent=2, ensure_ascii=False)}\n```\n\n"
                      f"## Candidate response\n\n{r['raw_response']}\n\n---\n")
    pd.DataFrame(sheet_rows).to_csv(out_dir / "calibration_sheet.csv", index=False)
    pd.DataFrame(key_rows).to_csv(out_dir / "calibration_key.csv", index=False)
    (out_dir / "calibration_bundle.md").write_text("\n".join(bundle))
    print(f"{len(sample)} samples -> {out_dir}/calibration_sheet.csv (+ calibration_bundle.md to read, calibration_key.csv withheld)")


def cmd_compare(args, cfg):
    out_dir = resolve(cfg, "results_dir") / "summary" / args.batch
    sheet = pd.read_csv(out_dir / "calibration_sheet.csv")
    key = pd.read_csv(out_dir / "calibration_key.csv")
    scored = {r["run_id"]: r for r in read_jsonl(resolve(cfg, "results_dir") / "scored" / args.batch / "scored.jsonl")}
    df = sheet.merge(key, on="sample_id")
    df = df[df["human_decision_quality"].notna()]
    if df.empty:
        raise SystemExit("no human-scored rows in calibration_sheet.csv")
    weights = cfg["scoring"]["weights"]
    rows = []
    for _, s in df.iterrows():
        j = scored[s["run_id"]]["judge"]
        human = JudgeResult(**{d: int(s[f"human_{d}"]) for d in DIMENSIONS},
                            critical_failure=bool(s["human_critical_failure"]),
                            critical_failure_type="ignored_decisive_evidence" if bool(s["human_critical_failure"]) else None,
                            expert_agreement=bool(s["human_expert_agreement"]), reasoning_summary="human")
        row = {"sample_id": s["sample_id"], "case_id": s["case_id"], "model_provider": s["model_provider"],
               "configuration": s["configuration"],
               "judge_score": weighted_score(JudgeResult(**j), weights), "human_score": weighted_score(human, weights)}
        for d in DIMENSIONS:
            row[f"diff_{d}"] = j[d] - getattr(human, d)
        row["cf_agree"] = bool(j["critical_failure"]) == human.critical_failure
        row["cf_judge_only"] = bool(j["critical_failure"]) and not human.critical_failure
        row["cf_human_only"] = human.critical_failure and not bool(j["critical_failure"])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["abs_score_diff"] = (out["judge_score"] - out["human_score"]).abs()
    out.to_csv(out_dir / "calibration_comparison.csv", index=False)

    lines = [f"# Judge calibration: batch {args.batch}", "", f"Samples: {len(out)}", "",
             f"Mean absolute weighted-score disagreement: {out['abs_score_diff'].mean():.1f}",
             f"Judge minus human mean score bias: {(out['judge_score'] - out['human_score']).mean():+.1f}",
             f"Critical-failure agreement: {out['cf_agree'].mean() * 100:.0f}% "
             f"(judge-only {int(out['cf_judge_only'].sum())}, human-only {int(out['cf_human_only'].sum())})", "",
             "## Per-dimension mean (judge - human) and mean absolute difference", ""]
    for d in DIMENSIONS:
        lines.append(f"- {d}: bias {out[f'diff_{d}'].mean():+.2f}, MAD {out[f'diff_{d}'].abs().mean():.2f}")
    lines += ["", "## By configuration", "",
              out.groupby("configuration")[["abs_score_diff", "cf_agree"]].mean().round(2).to_markdown(), ""]
    (out_dir / "calibration_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--batch", required=True); e.add_argument("--fraction", type=float, default=0.15); e.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("compare"); c.add_argument("--batch", required=True)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    (cmd_export if args.cmd == "export" else cmd_compare)(args, cfg)


if __name__ == "__main__":
    main()
