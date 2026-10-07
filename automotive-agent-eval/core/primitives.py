"""Specialised cognitive primitives (implementation C: general agent delegating to a primitive).

A primitive is a cheap, deterministic function that pre-processes the candidate-visible evidence
and appends its output to the payload, so the general model reasons over the primitive's result
instead of doing the work itself. Each primitive reports its own cost and latency, which are added
to the run's economics.

These are REFERENCE primitives: simple, transparent baselines that establish whether delegation
changes the economic frontier at all. A production primitive (learned ranker, valuation model,
entity resolver) would replace the function body; the interface and accounting stay the same.
"""
from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from core.models import Case, CandidateVisible


@dataclass
class PrimitiveOutput:
    name: str
    primitive: str                       # cognitive primitive class, e.g. "rank"
    summary: dict[str, Any]
    latency_ms: int
    cost: float = 0.0
    visible: CandidateVisible | None = None


Registry: dict[str, Callable[[Case], PrimitiveOutput]] = {}


def register(name: str):
    def deco(fn):
        Registry[name] = fn
        return fn
    return deco


def apply_primitives(case: Case, names: list[str]) -> tuple[CandidateVisible, list[PrimitiveOutput]]:
    visible = deepcopy(case.candidate_visible)
    outputs: list[PrimitiveOutput] = []
    for name in names:
        fn = Registry[name]
        tmp_case = case.model_copy(update={"candidate_visible": visible})
        out = fn(tmp_case)
        visible = out.visible or visible
        visible.evidence.setdefault("specialist_outputs", []).append({"engine": name, **out.summary})
        outputs.append(out)
    return visible, outputs


# ----------------------------------------------------------------------------- helpers

def _listings(v: CandidateVisible) -> tuple[str, list[dict]] | tuple[None, None]:
    for key in ("listings", "alternatives", "comparables"):
        items = v.evidence.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict) and "id" in items[0]:
            return key, items
    return None, None


def _num(x) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"[\d,]+(?:\.\d+)?", x)
        if m:
            return float(m.group(0).replace(",", ""))
    return None


def _text(d: dict) -> str:
    return " ".join(str(v) for v in d.values()).lower()


# -------------------------------------------------------------------------- primitives

@register("hard_constraint_filter")
def hard_constraint_filter(case: Case) -> PrimitiveOutput:
    """filter: drop listings that violate budget, gearbox, fuel, seat or body hard requirements
    stated in user_context. Conservative: only removes on explicit evidence."""
    start = time.perf_counter()
    v = case.candidate_visible
    key, items = _listings(v)
    if not items:
        return PrimitiveOutput("hard_constraint_filter", "filter", {"note": "no listing set"}, 0)
    uc = v.user_context
    budget = None
    for k, val in uc.items():
        if "budget" in k.lower():
            budget = _num(val)
    reqs = " ".join(str(x) for x in uc.get("hard_requirements", [])).lower()
    kept, dropped = [], []
    for it in items:
        reasons = []
        price = _num(it.get("price_gbp") or it.get("price") or it.get("price_gbp_cash"))
        if budget and price and price > budget:
            reasons.append(f"price {int(price)} > budget {int(budget)}")
        trans = str(it.get("transmission", "")).lower()
        if "automatic" in reqs and trans and "auto" not in trans and "dsg" not in trans and "cvt" not in trans and "edc" not in trans and "dct" not in trans:
            reasons.append("manual gearbox")
        fuel = str(it.get("fuel", "")).lower()
        if ("no home charging" in reqs or "petrol or hybrid" in reqs) and fuel and ("electric" in fuel and "hybrid" not in fuel):
            reasons.append("battery electric")
        seats = it.get("seats")
        m = re.search(r"(\d)\s*(?:\+)?\s*seats?|seven[- ]seat|7[- ]seat", reqs)
        if m and seats and m.group(1) and int(seats) < int(m.group(1)):
            reasons.append(f"{seats} seats")
        (dropped if reasons else kept).append((it["id"], reasons))
    new_v = deepcopy(v)
    new_v.evidence[key] = [it for it in items if it["id"] in {k for k, _ in kept}]
    new_v.evidence[f"{key}_removed_by_filter"] = [{"id": i, "reasons": r} for i, r in dropped]
    summary = {"primitive": "filter", "input_count": len(items), "kept": len(kept), "removed": len(dropped),
               "removed_ids": [i for i, _ in dropped],
               "note": "Hard-constraint filter from stated user requirements. Listings removed are shown with reasons; they can be reinstated if a requirement is soft."}
    return PrimitiveOutput("hard_constraint_filter", "filter", summary, int((time.perf_counter() - start) * 1000), 0.0, new_v)


@register("heuristic_ranker")
def heuristic_ranker(case: Case, top_k: int = 8) -> PrimitiveOutput:
    """rank: transparent weighted score over price headroom, mileage, owners, history evidence,
    warranty, equipment count. Produces a ranked top-k with the score breakdown."""
    start = time.perf_counter()
    v = case.candidate_visible
    key, items = _listings(v)
    if not items:
        return PrimitiveOutput("heuristic_ranker", "rank", {"note": "no listing set"}, 0)
    prices = [p for p in (_num(i.get("price_gbp") or i.get("price")) for i in items) if p]
    miles = [m for m in (_num(i.get("mileage")) for i in items) if m]
    pmin, pmax = (min(prices), max(prices)) if prices else (0, 1)
    mmin, mmax = (min(miles), max(miles)) if miles else (0, 1)

    def norm(x, lo, hi):
        return 0.5 if x is None or hi == lo else (x - lo) / (hi - lo)

    ranked = []
    for it in items:
        t = _text(it)
        s = {}
        s["price"] = 1 - norm(_num(it.get("price_gbp") or it.get("price")), pmin, pmax)
        s["mileage"] = 1 - norm(_num(it.get("mileage")), mmin, mmax)
        owners = _num(it.get("previous_owners"))
        s["owners"] = 1.0 if owners is None else max(0.0, 1 - (owners - 1) * 0.25)
        s["history"] = 1.0 if "full" in str(it.get("service_history", "")).lower() else 0.4 if "part" in str(it.get("service_history", "")).lower() else 0.5
        s["history"] *= 0.8 if "claimed" in str(it.get("service_history", "")).lower() or "not yet seen" in t else 1.0
        s["warranty"] = 1.0 if re.search(r"warranty.{0,40}(20(2[7-9]|3\d)|12 months|manufacturer)", t) else 0.5
        s["risk"] = 0.3 if re.search(r"advisor(y|ies).{0,80}(corros|leak|bush|worn|play)", t) else 1.0
        eq = it.get("notable_equipment") or it.get("equipment") or []
        s["equipment"] = min(len(eq) / 8, 1.0) if isinstance(eq, list) else 0.5
        total = (0.25 * s["price"] + 0.15 * s["mileage"] + 0.10 * s["owners"] + 0.20 * s["history"]
                 + 0.10 * s["warranty"] + 0.10 * s["risk"] + 0.10 * s["equipment"])
        ranked.append({"id": it["id"], "score": round(total, 3), "components": {k: round(x, 2) for k, x in s.items()}})
    ranked.sort(key=lambda r: -r["score"])
    top = ranked[:top_k]
    new_v = deepcopy(v)
    order = {r["id"]: i for i, r in enumerate(ranked)}
    new_v.evidence[key] = sorted(items, key=lambda it: order[it["id"]])[:top_k]
    new_v.evidence[f"{key}_below_cutoff_ids"] = [r["id"] for r in ranked[top_k:]]
    summary = {"primitive": "rank", "input_count": len(items), "returned": len(top), "ranking": top,
               "weights": {"price": 0.25, "mileage": 0.15, "owners": 0.10, "history": 0.20, "warranty": 0.10, "risk": 0.10, "equipment": 0.10},
               "note": "Heuristic ranking engine output. Scores are relative within this set, not valuations. The full listing detail is provided only for the returned items."}
    return PrimitiveOutput("heuristic_ranker", "rank", summary, int((time.perf_counter() - start) * 1000), 0.0, new_v)


@register("duplicate_detector")
def duplicate_detector(case: Case) -> PrimitiveOutput:
    """normalise / detect_anomaly: cluster listings that are probably the same physical vehicle
    (same make/model/year/colour and MOT expiry, or shared partial plate, with mileage within 1%)."""
    start = time.perf_counter()
    v = case.candidate_visible
    key, items = _listings(v)
    if not items:
        return PrimitiveOutput("duplicate_detector", "normalise", {"note": "no listing set"}, 0)
    clusters: list[list[dict]] = []
    for it in items:
        placed = False
        for cl in clusters:
            ref = cl[0]
            same_core = all(str(it.get(k, "")).lower() == str(ref.get(k, "")).lower() for k in ("make", "model", "year") if k in it and k in ref)
            c1, c2 = set(str(it.get("colour", "")).lower().split()), set(str(ref.get("colour", "")).lower().split())
            same_core = same_core and (not c1 or not c2 or bool(c1 & c2 - {"metallic", "pearl", "solid"}))
            plate = it.get("registration") or it.get("plate_partial") or it.get("reg")
            rplate = ref.get("registration") or ref.get("plate_partial") or ref.get("reg")
            same_plate = bool(plate and rplate and str(plate)[:4].upper() == str(rplate)[:4].upper())
            same_mot = it.get("mot_expiry") and it.get("mot_expiry") == ref.get("mot_expiry")
            m1, m2 = _num(it.get("mileage")), _num(ref.get("mileage"))
            close_miles = m1 and m2 and abs(m1 - m2) / max(m1, m2) < 0.01
            if same_core and close_miles and (same_plate or same_mot):
                cl.append(it); placed = True; break
        if not placed:
            clusters.append([it])
    dupes = [[i["id"] for i in cl] for cl in clusters if len(cl) > 1]
    summary = {"primitive": "normalise", "input_count": len(items), "distinct_vehicles": len(clusters),
               "probable_duplicates": dupes,
               "note": "Entity-resolution engine output: listings grouped as the same physical vehicle on matching identity fields and mileage within 1%. Treat as probable, not certain."}
    return PrimitiveOutput("duplicate_detector", "normalise", summary, int((time.perf_counter() - start) * 1000), 0.0, None)


@register("comparable_valuation")
def comparable_valuation(case: Case) -> PrimitiveOutput:
    """estimate: median of comparables adjusted for mileage (£/1,000 miles) and trim/owner differences
    with an explicit, crude adjustment table. Emits a range, never a point."""
    start = time.perf_counter()
    v = case.candidate_visible
    comps = v.evidence.get("comparables")
    target = v.evidence.get("target_vehicle")
    if not (isinstance(comps, list) and comps and isinstance(target, dict)):
        return PrimitiveOutput("comparable_valuation", "estimate", {"note": "no target/comparables"}, 0)
    tprice, tmiles = _num(target.get("price_gbp")), _num(target.get("mileage"))
    adj = []
    per_1k = 0.0025  # 0.25% of price per 1,000 miles difference
    for c in comps:
        p, m = _num(c.get("price_gbp")), _num(c.get("mileage"))
        if not p:
            continue
        a = p
        notes = []
        if m and tmiles:
            delta = (m - tmiles) / 1000 * per_1k * p
            a += delta; notes.append(f"mileage {delta:+.0f}")
        ct, tt = str(c.get("trim", "")).lower(), str(target.get("derivative", "") or target.get("trim", "")).lower()
        if ct and tt and ct not in tt:
            a += 0.04 * p; notes.append("trim +4%")
        co, to = _num(c.get("previous_owners")), _num(target.get("previous_owners"))
        if co and to and co > to + 1:
            a += 0.02 * p; notes.append("owners +2%")
        if "part" in str(c.get("service_history", "")).lower():
            a += 0.03 * p; notes.append("history +3%")
        adj.append({"id": c["id"], "asking": p, "adjusted": round(a), "adjustments": notes})
    vals = sorted(x["adjusted"] for x in adj)
    if not vals:
        return PrimitiveOutput("comparable_valuation", "estimate", {"note": "no priced comparables"}, 0)
    lo, hi, med = vals[0], vals[-1], vals[len(vals) // 2]
    summary = {"primitive": "estimate", "target_asking": tprice, "adjusted_comparables": adj,
               "indicative_range_gbp": [int(med * 0.97), int(med * 1.03)], "adjusted_median_gbp": int(med),
               "position_vs_adjusted_median_gbp": None if tprice is None else int(tprice - med),
               "note": "Comparable-adjusted indicative range from asking prices only. Not a transaction-price valuation; adjustment factors are fixed heuristics."}
    return PrimitiveOutput("comparable_valuation", "estimate", summary, int((time.perf_counter() - start) * 1000), 0.0, None)
