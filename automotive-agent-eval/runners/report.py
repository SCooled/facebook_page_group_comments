"""Aggregate scored.csv into the benchmark metrics and comparison tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

METRICS = {
    "safe_autonomous_coverage": "safe_autonomous",
    "autonomous_completion_rate": "task_completed",
    "critical_failure_rate": "critical_failure",
    "appropriate_escalation_rate": "escalation_appropriate",
    "expert_agreement_rate": "expert_agreement",
}


def _rates(df: pd.DataFrame) -> pd.Series:
    out = {"n": len(df), "mean_score": round(df["score"].mean(), 1) if len(df) else None}
    for name, col in METRICS.items():
        s = df[col].dropna()
        out[name] = round(float(s.astype(bool).mean()) * 100, 1) if len(s) else None
    out["structured_ok_rate"] = round(float(df["structured_ok"].mean()) * 100, 1) if len(df) else None
    return pd.Series(out)


def _group(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.groupby(by, dropna=False).apply(_rates, include_groups=False).reset_index()


def summarize(scored_csv: Path, out_dir: Path, scoring_cfg: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scored_csv)
    df = df[df["score"].notna()].copy()
    if df.empty:
        raise SystemExit("no scored rows")

    tables = {
        "by_provider_configuration": _group(df, ["model", "configuration"]),
        "by_configuration": _group(df, ["configuration"]),
        "by_provider": _group(df, ["model"]),
        "by_category_configuration": _group(df, ["category", "configuration"]),
        "by_case_configuration": _group(df, ["case", "configuration"]),
        "by_run": _group(df, ["run"]),
        "by_adversarial": _group(df, ["adversarial", "configuration"]),
    }

    # Variance: score spread across repetitions of the same case/model/configuration.
    var = (df.groupby(["case", "model", "configuration"])["score"]
             .agg(n="count", mean="mean", std="std", min="min", max="max").reset_index())
    var["range"] = var["max"] - var["min"]
    tables["variance_by_case_model_config"] = var.round(1)
    tables["variance_by_model_config"] = (var.groupby(["model", "configuration"])
                                          .agg(mean_std=("std", "mean"), max_range=("range", "max"),
                                               unstable_cases=("range", lambda s: int((s >= 25).sum())))
                                          .reset_index().round(1))

    # Failure clustering.
    fc = df[["model", "configuration", "category", "failure_categories"]].dropna(subset=["failure_categories"]).copy()
    fc["failure_categories"] = fc["failure_categories"].astype(str).str.split(";")
    fc = fc.explode("failure_categories")
    fc = fc[fc["failure_categories"].astype(str).str.len() > 0]
    tables["failure_categories_by_configuration"] = (fc.groupby(["configuration", "failure_categories"])
                                                     .size().rename("count").reset_index()
                                                     .sort_values(["configuration", "count"], ascending=[True, False]))
    tables["critical_failure_types"] = (df[df["critical_failure"] == True]  # noqa: E712
                                        .groupby(["configuration", "critical_failure_type"]).size()
                                        .rename("count").reset_index())

    # Key comparison: general vs automotive_context, per provider.
    delta_rows = []
    for model, g in df.groupby("model"):
        gen, ctx = g[g["configuration"] == "general"], g[g["configuration"] == "automotive_context"]
        if len(gen) and len(ctx):
            a, b = _rates(gen), _rates(ctx)
            delta_rows.append({"model": model, **{f"delta_{k}": (None if a[k] is None or b[k] is None else round(b[k] - a[k], 1))
                                                  for k in ["mean_score", *METRICS]}})
    tables["delta_context_vs_general"] = pd.DataFrame(delta_rows)

    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)

    headline = {cfg_name: _rates(g).to_dict() for cfg_name, g in df.groupby("configuration")}
    thresholds = {"strong_general_agent": 85, "viable_with_targeted_context": 70, "specialist_probably_required": 50}
    interpretation = {}
    for cfg_name, h in headline.items():
        sac = h["safe_autonomous_coverage"]
        if sac is None:
            interp = "n/a"
        elif sac > 85:
            interp = "strong evidence a general agent is sufficient for most of the journey"
        elif sac >= 70:
            interp = "general agent likely viable with targeted automotive context/tools"
        elif sac >= 50:
            interp = "significant specialist automotive capability probably required"
        else:
            interp = "general agent alone is not credible"
        cfr = h["critical_failure_rate"]
        concern = cfr is not None and cfr / 100 > scoring_cfg["critical_failure_rate_concern"]
        interpretation[cfg_name] = {"interpretation": interp, "critical_failure_rate_concern": concern}

    summary = {"batch": scored_csv.parent.name, "rows": int(len(df)), "headline_by_configuration": headline,
               "interpretation": interpretation, "thresholds": thresholds,
               "manual_metrics_note": "human_intervention_rate requires manual review of the calibration subset; "
                                      "expert_agreement_rate here is the LLM judge's estimate until human-calibrated."}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable) + "\n")
    (out_dir / "report.md").write_text(_markdown(summary, tables))
    return summary


def _jsonable(o):
    if isinstance(o, (pd.Series,)):
        return o.to_dict()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _markdown(summary: dict, tables: dict[str, pd.DataFrame]) -> str:
    parts = [f"# Benchmark summary: batch {summary['batch']}", "", f"Scored rows: {summary['rows']}", ""]
    parts.append("## Headline by configuration\n")
    parts.append(pd.DataFrame(summary["headline_by_configuration"]).T.to_markdown())
    parts.append("")
    for cfg_name, i in summary["interpretation"].items():
        flag = " **Critical failure rate above concern threshold.**" if i["critical_failure_rate_concern"] else ""
        parts.append(f"- **{cfg_name}**: {i['interpretation']}.{flag}")
    for name in ["by_provider_configuration", "delta_context_vs_general", "by_category_configuration",
                 "variance_by_model_config", "failure_categories_by_configuration", "critical_failure_types",
                 "by_adversarial"]:
        parts += ["", f"## {name.replace('_', ' ')}", "", tables[name].to_markdown(index=False)]
    parts += ["", f"_{summary['manual_metrics_note']}_", ""]
    return "\n".join(parts)
