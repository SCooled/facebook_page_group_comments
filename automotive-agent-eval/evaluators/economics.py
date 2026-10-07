"""Economic scoring: cost, latency, reliability, throughput, utility and economic specialist advantage.

Per run:
  model_cost  = input_tokens * price_in + output_tokens * price_out   (per-token prices from config)
  total_cost  = model_cost + primitive_cost
  quality Q   = weighted_score / 100, zeroed when the run is a critical failure

Per implementation (provider x configuration), aggregated over runs:
  Q  mean quality
  C  mean total_cost per task, and cost per safe completion (total cost / safe completions)
  L  p50 and p95 total latency
  R  reliability = P(valid and not critical failure); score std across repetitions reported alongside
  T  throughput = objects_in_scope per second of latency (objects/s) and tasks per hour at configured workers

Utility U = w_q*Q + w_c*C' + w_l*L' + w_r*R + w_t*T'   (all terms in [0,1], weights sum to 1, so U in [0,1])
  C' = ref_c / (ref_c + C)        cheaper is better; a task at the reference cost scores 0.5
  L' = ref_l / (ref_l + L)        faster is better; at the reference latency scores 0.5
  T' = T / (T + ref_t)            more objects per second is better; at the reference rate scores 0.5
  The saturating transforms stop any single dimension (e.g. millisecond latency) swamping the others.
  This is the brief's U = w_q Q - w_c C - w_l L + w_r R + w_t T with cost and latency expressed as
  bounded "goodness" scores rather than unbounded penalties.
ESA (economic specialist advantage) = U_candidate - U_baseline for the same provider, baseline = general.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def model_cost(model_name: str, provider: str, input_tokens: int | None, output_tokens: int | None,
               pricing: dict[str, Any]) -> float | None:
    """Price lookup by exact model name, then by prefix, then by provider default."""
    table = pricing.get("models", {})
    entry = table.get(model_name)
    if entry is None:
        for key, val in table.items():
            if model_name and model_name.startswith(key):
                entry = val
                break
    if entry is None:
        entry = pricing.get("provider_defaults", {}).get(provider)
    if entry is None:
        return None
    if "per_task" in entry:                       # e.g. human operator priced per task
        return float(entry["per_task"])
    if input_tokens is None or output_tokens is None:
        return None
    return round(input_tokens / 1e6 * entry["input_per_million"] + output_tokens / 1e6 * entry["output_per_million"], 6)


def utility(q: float, c: float, l_ms: float, r: float, t: float, weights: dict[str, float], ref: dict[str, float]) -> float:
    c_good = ref["cost_per_task"] / (ref["cost_per_task"] + max(c, 0))
    l_good = ref["latency_ms"] / (ref["latency_ms"] + max(l_ms, 0))
    t_good = t / (t + ref["objects_per_second"]) if t > 0 else 0.0
    return round(weights["quality"] * q + weights["cost"] * c_good + weights["latency"] * l_good
                 + weights["reliability"] * r + weights["throughput"] * t_good, 4)


def implementation_vector(df: pd.DataFrame, econ_cfg: dict[str, Any], workers: int) -> pd.Series:
    """Aggregate one group of scored rows into the (Q, C, L, R, T) vector plus utilities."""
    n = len(df)
    valid = df["valid"].astype(bool)
    crit = df["critical_failure"].astype(bool)
    safe = df["safe_autonomous"].fillna(False).astype(bool)
    q = df["quality"].fillna(0).mean()
    cost = df["total_cost"].fillna(0).mean()
    safe_n = int(safe.sum())
    cost_per_safe = (df["total_cost"].fillna(0).sum() / safe_n) if safe_n else None
    lat = df["total_latency_ms"].astype(float)
    p50, p95 = lat.quantile(0.5), lat.quantile(0.95)
    r = float((valid & ~crit).mean())
    obj_per_s = float((df["objects_in_scope"] / (lat.clip(lower=1) / 1000)).mean())
    tasks_per_hour = (workers * 3_600_000 / p50) if p50 else None
    std = df.groupby("case")["score"].std().mean() if "case" in df else None
    out = {
        "n": n, "Q_quality": round(q, 3), "mean_score": round(df["score"].mean(), 1),
        "safe_autonomous_coverage": round(safe.mean() * 100, 1),
        "C_cost_per_task": round(cost, 4), "cost_per_safe_completion": None if cost_per_safe is None else round(cost_per_safe, 4),
        "L_p50_ms": int(p50), "L_p95_ms": int(p95),
        "R_reliability": round(r, 3), "critical_failure_rate": round(crit.mean() * 100, 1),
        "invalid_rate": round((~valid).mean() * 100, 1),
        "score_std_within_case": None if std is None or pd.isna(std) else round(std, 1),
        "T_objects_per_second": round(obj_per_s, 3),
        "T_tasks_per_hour": None if tasks_per_hour is None else int(tasks_per_hour),
        "cost_per_object": round(cost / max(df["objects_in_scope"].mean(), 1), 5),
    }
    for name, w in econ_cfg["weight_profiles"].items():
        out[f"U_{name}"] = utility(q, cost, p50, r, obj_per_s, w, econ_cfg["reference_scales"])
    return pd.Series(out)


def esa_table(vectors: pd.DataFrame, baseline_configuration: str, profiles: list[str]) -> pd.DataFrame:
    """Economic specialist advantage of every (provider, configuration) against the same provider's baseline."""
    rows = []
    for model, g in vectors.groupby("model"):
        base = g[g["configuration"] == baseline_configuration]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, row in g.iterrows():
            if row["configuration"] == baseline_configuration:
                continue
            rec = {"model": model, "implementation": row["configuration"], "baseline": baseline_configuration,
                   "dQ": round(row["Q_quality"] - b["Q_quality"], 3),
                   "dC_cost_per_task": round(row["C_cost_per_task"] - b["C_cost_per_task"], 4),
                   "dL_p50_ms": int(row["L_p50_ms"] - b["L_p50_ms"]),
                   "dR": round(row["R_reliability"] - b["R_reliability"], 3),
                   "dT_objects_per_second": round(row["T_objects_per_second"] - b["T_objects_per_second"], 3)}
            for p in profiles:
                rec[f"ESA_{p}"] = round(row[f"U_{p}"] - b[f"U_{p}"], 4)
            rows.append(rec)
    return pd.DataFrame(rows)
