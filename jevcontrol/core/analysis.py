"""Turns saved per-task rows into the numbers the report shows.

Everything is *paired*: every arm ran the same tasks, so accuracy/latency deltas are computed per task and the
uncertainty comes from a paired bootstrap, not from comparing two independent means.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .types import ExperimentConfig

MAX_SWEEP_POINTS = 80


def _pct(x: list[float], q: float) -> float:
    return float(np.percentile(x, q)) if x else 0.0


def _mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else 0.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant pairs b (cand better) vs c (base better)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_bootstrap(diff: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    if len(diff) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ratio_bootstrap(num: np.ndarray, den: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    """CI of mean(num)/mean(den) resampling task pairs."""
    if len(num) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    r = num[idx].mean(axis=1) / np.maximum(den[idx].mean(axis=1), 1e-9)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# ---------------------------------------------------------------------------------------------------------

def arm_summary(arm: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    n = len(rows)
    decs = [d for r in rows for d in r.get("decisions", [])]
    truthed = [d for d in decs if d.get("correct") is not None]
    e2e = [r["e2e_ms"] for r in rows]
    per = lambda key: _mean([r[key] for r in rows])
    return {
        "id": arm.id, "label": arm.title(), "kind": arm.kind, "tau": arm.tau,
        "decider": arm.decider.label() if arm.decider else None,
        "n": n, "errors": n - len(ok),
        "accuracy": _mean([r["score"] for r in rows]),
        "e2e_ms": {"mean": _mean(e2e), "p50": _pct(e2e, 50), "p95": _pct(e2e, 95)},
        "llm_calls": per("llm_calls"), "llm_generate_calls": per("llm_generate_calls"),
        "llm_decide_calls": per("llm_decide_calls"),
        "llm_prompt_tokens": per("llm_prompt_tokens"), "llm_completion_tokens": per("llm_completion_tokens"),
        "decider_calls": per("decider_calls"), "decider_ms": per("decider_ms"),
        "cost_per_1k": per("cost_usd") * 1000,
        "decisions_per_task": len(decs) / n if n else 0.0,
        "offload_rate": (sum(1 for d in decs if d["source"] == "menu") / len(decs)) if decs else 0.0,
        "escalation_rate": (sum(1 for d in decs if d.get("escalated")) / len(decs)) if decs else 0.0,
        "parse_fail_rate": (sum(1 for d in decs if not d.get("parsed", True)) / len(decs)) if decs else 0.0,
        "decision_accuracy": _mean([1.0 if d["correct"] else 0.0 for d in truthed]) if truthed else None,
        "decision_truth_n": len(truthed),
        "label_mass": _mean([d["label_mass"] for d in decs if d.get("label_mass") is not None]) if any(
            d.get("label_mass") is not None for d in decs) else None,
    }


def paired(cfg: ExperimentConfig, base_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    b_by = {r["task_id"]: r for r in base_rows}
    common = [(b_by[r["task_id"]], r) for r in rows if r["task_id"] in b_by]
    if not common:
        return {"n": 0}
    sb = np.array([b["score"] for b, _ in common], float)
    sc = np.array([c["score"] for _, c in common], float)
    diff = sc - sb
    lo, hi = paired_bootstrap(diff, cfg.bootstrap, cfg.seed)
    wins = int((sc > sb).sum())
    losses = int((sc < sb).sum())
    eb = np.array([b["e2e_ms"] for b, _ in common], float)
    ec = np.array([c["e2e_ms"] for _, c in common], float)
    s_lo, s_hi = ratio_bootstrap(eb, ec, cfg.bootstrap, cfg.seed)
    tb = np.array([b["llm_prompt_tokens"] + b["llm_completion_tokens"] for b, _ in common], float)
    tc = np.array([c["llm_prompt_tokens"] + c["llm_completion_tokens"] for _, c in common], float)
    cb = np.array([b["llm_calls"] for b, _ in common], float)
    cc = np.array([c["llm_calls"] for _, c in common], float)
    kb = np.array([b["cost_usd"] for b, _ in common], float)
    kc = np.array([c["cost_usd"] for _, c in common], float)
    m = cfg.margin
    delta = float(diff.mean())
    # Non-inferiority: safe when even the pessimistic end of the interval stays within the margin; worse when the
    # loss is both beyond the margin and statistically detectable (interval entirely below zero); else not proven.
    var = float(np.var(diff, ddof=1)) if len(diff) > 1 else 0.0
    tasks_needed = math.ceil(3.8416 * var / (m * m)) if m > 0 else None  # n for a 95% CI half-width of m
    verdict = "safe" if lo >= -m else ("worse" if (delta < -m and hi < 0) else "unclear")
    return {
        "n": len(common),
        "delta_acc": delta, "ci": [lo, hi], "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
        "mcnemar_p": mcnemar_exact(wins, losses),
        "speedup": float(eb.mean() / max(ec.mean(), 1e-9)), "speedup_ci": [s_lo, s_hi],
        "token_reduction": float(1 - tc.mean() / tb.mean()) if tb.mean() > 0 else 0.0,
        "calls_reduction": float(1 - cc.mean() / cb.mean()) if cb.mean() > 0 else 0.0,
        "cost_reduction": float(1 - kc.mean() / kb.mean()) if kb.mean() > 0 else None,
        "verdict": verdict, "margin": m, "leans_worse": bool(delta < -m), "tasks_needed": tasks_needed,
        "flips": [{"task_id": c["task_id"], "base": b["score"], "cand": c["score"]}
                  for b, c in common if b["score"] != c["score"]],
    }


def _matched(base_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
    """Decisions asked identically (same task, site, key, state, question) under the baseline and the candidate."""
    idx: dict[tuple[str, str], dict] = {}
    for r in base_rows:
        for d in r.get("decisions", []):
            idx.setdefault((r["task_id"], d["fingerprint"]), d)
    out, seen = [], set()
    for r in rows:
        for d in r.get("decisions", []):
            k = (r["task_id"], d["fingerprint"])
            if k in idx and k not in seen:
                seen.add(k)
                out.append((idx[k], d))
    return out


def _cand_view(d: dict[str, Any]) -> tuple[str, float]:
    """The menu model's own answer + confidence (also for hybrid decisions that were escalated)."""
    if d.get("menu_selected") is not None:
        return d["menu_selected"], float(d["menu_confidence"] or 0.0)
    return d["selected"], float(d["confidence"] or 0.0)


def _sweep(pairs: list[tuple[dict, dict]]) -> list[dict[str, Any]]:
    """Coverage/quality curve over matched decisions, one point per distinct confidence value.

    A point (tau, offload, ...) means: keep the decision model's answer where its confidence >= tau and use the
    baseline LLM's answer elsewhere. Built from the data itself, so it is exact for over-confident models too
    (whose useful thresholds all sit between 0.99 and 1.0, where a fixed grid would see nothing).
    Returned ascending in tau, so points[0] is "the decision model answers everything".
    """
    n = len(pairs)
    if not n:
        return []
    items = []
    for b, c in pairs:
        sel, conf = _cand_view(c)
        t = c.get("truth")
        items.append({"conf": conf, "agree": sel == b["selected"],
                      "cand_ok": (sel == t) if t is not None else None, "base_ok": (b["selected"] == t) if t is not None else None})
    items.sort(key=lambda x: -x["conf"])
    truth_items = [i for i in items if i["cand_ok"] is not None]
    n_truth = len(truth_items)
    base_total = sum(1 for i in truth_items if i["base_ok"])
    base_acc = base_total / n_truth if n_truth else None

    def point(tau: float, k: int, agree: int, cand_ok: int, base_ok_handled: int) -> dict[str, Any]:
        pt: dict[str, Any] = {"tau": tau, "offload": k / n, "agreement": agree / k if k else None, "n": n}
        if n_truth:
            pt["hybrid_acc"] = (cand_ok + base_total - base_ok_handled) / n_truth
            pt["base_acc"] = base_acc
            pt["truth_n"] = n_truth
        return pt

    pts = [point(1.0001, 0, 0, 0, 0)]  # the decision model handles nothing
    k = agree = cand_ok = base_ok_handled = 0
    idx = 0
    while idx < n:
        conf = items[idx]["conf"]
        while idx < n and items[idx]["conf"] == conf:
            it = items[idx]
            k += 1
            agree += it["agree"]
            if it["cand_ok"] is not None:
                cand_ok += it["cand_ok"]
                base_ok_handled += it["base_ok"]
            idx += 1
        pts.append(point(conf, k, agree, cand_ok, base_ok_handled))
    if len(pts) > MAX_SWEEP_POINTS:
        keep = sorted({round(i * (len(pts) - 1) / (MAX_SWEEP_POINTS - 1)) for i in range(MAX_SWEEP_POINTS)})
        pts = [pts[i] for i in keep]
    return list(reversed(pts))


def _recommend(points: list[dict[str, Any]], min_offloaded: int = 5) -> dict[str, Any] | None:
    """Lowest threshold (= most offloading) that keeps decision quality: hybrid accuracy within 1 point of the
    baseline LLM's when ground truth exists, else >= 97% agreement with the baseline LLM on offloaded decisions."""
    for p in points:
        if p["offload"] * p["n"] < min_offloaded:
            continue
        ok = (p["hybrid_acc"] >= p["base_acc"] - 0.01) if "hybrid_acc" in p else (p["agreement"] or 0) >= 0.97
        if ok:
            return {"tau": p["tau"], "offload": p["offload"], "agreement": p["agreement"],
                    "basis": "ground truth" if "hybrid_acc" in p else "agreement with the LLM"}
    return None


def sites_and_sweeps(base_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    matched = _matched(base_rows, rows)
    by_site_all: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for d in r.get("decisions", []):
            by_site_all[d["site"]].append(d)
    base_site: dict[str, list[dict]] = defaultdict(list)
    for r in base_rows:
        for d in r.get("decisions", []):
            base_site[d["site"]].append(d)
    by_site_pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for b, c in matched:
        by_site_pairs[c["site"]].append((b, c))
    sites = []
    for site, ds in sorted(by_site_all.items()):
        pairs = by_site_pairs.get(site, [])
        truthed = [d for d in ds if d.get("correct") is not None]
        btruth = [d for d in base_site.get(site, []) if d.get("correct") is not None]
        sites.append({
            "site": site, "kind": ds[0]["kind"], "n": len(ds),
            "offload": _mean([1.0 if d["source"] == "menu" else 0.0 for d in ds]),
            "mean_confidence": _mean([_cand_view(d)[1] for d in ds if d.get("confidence") is not None or d.get("menu_confidence") is not None]),
            "latency_ms": _mean([d["latency_ms"] for d in ds]),
            "base_latency_ms": _mean([d["latency_ms"] for d in base_site.get(site, [])]),
            "agreement": _mean([1.0 if _cand_view(c)[0] == b["selected"] else 0.0 for b, c in pairs]) if pairs else None,
            "agreement_n": len(pairs),
            "truth_acc": _mean([1.0 if d["correct"] else 0.0 for d in truthed]) if truthed else None,
            "base_truth_acc": _mean([1.0 if d["correct"] else 0.0 for d in btruth]) if btruth else None,
            "truth_n": len(truthed),
        })
    overall = _sweep(matched)
    sweeps = {"overall": {"points": overall, "recommended": _recommend(overall), "n": len(matched)}, "sites": {}}
    for site, pairs in sorted(by_site_pairs.items()):
        pts = _sweep(pairs)
        sweeps["sites"][site] = {"points": pts, "recommended": _recommend(pts), "n": len(pairs)}
    return sites, sweeps


def callmap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The harness's steps in the order they run, and for each: who answered it in this arm.

    This is the "what did we actually move" view: a decision step answered by the decision model emits one
    token; the same step in the baseline is a prompted LLM call; a generation step is an LLM call in every arm.
    """
    n = max(len(rows), 1)
    agg: dict[str, dict[str, Any]] = {}
    positions: dict[str, list[int]] = {}
    for r in rows:
        for i, c in enumerate(r.get("calls", [])):
            site = c.get("site") or "(generation)"
            a = agg.setdefault(site, {"site": site, "calls": 0, "prompt_tokens": 0, "out_tokens": 0,
                                      "latency": [], "roles": set(), "sources": set(), "kind": None})
            a["calls"] += 1
            a["prompt_tokens"] += c["prompt_tokens"]
            a["out_tokens"] += c["completion_tokens"]
            a["latency"].append(c["latency_ms"])
            a["roles"].add(c["role"])
            a["sources"].add("llm" if c["on_main_llm"] else "decision model")
            positions.setdefault(site, []).append(i)
    for r in rows:  # decisions carry the primitive, and the decider's own latency for menu answers
        for d in r.get("decisions", []):
            a = agg.get(d["site"])
            if a is not None:
                a["kind"] = d["kind"]
                a["sources"].add("decision model" if d["source"] == "menu" else "llm")
    out = []
    for site, a in agg.items():
        src = a["sources"]
        out.append({
            "site": site, "kind": a["kind"] or "generation",
            "role": "decide" if "decide" in a["roles"] or a["kind"] else "generate",
            "answered_by": "decision model" if src == {"decision model"} else ("llm" if src == {"llm"} else "mixed"),
            "calls_per_task": a["calls"] / n, "calls": a["calls"],
            "prompt_tokens_per_task": a["prompt_tokens"] / n, "out_tokens_per_task": a["out_tokens"] / n,
            "med_latency_ms": float(np.median(a["latency"])) if a["latency"] else 0.0,
            "pos": float(np.median(positions.get(site, [0]))),
        })
    return sorted(out, key=lambda x: (x["pos"], x["site"]))


def analyze(cfg: ExperimentConfig, rows_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = next((a for a in cfg.arms if a.kind == "baseline"), None)
    out: dict[str, Any] = {"arms": [], "paired": {}, "sites": {}, "sweeps": {}, "callmap": {},
                           "baseline": base.id if base else None, "margin": cfg.margin}
    for arm in cfg.arms:
        rows = rows_by_arm.get(arm.id, [])
        if not rows:
            continue
        out["arms"].append(arm_summary(arm, rows))
        out["callmap"][arm.id] = callmap(rows)
        if base and arm.id != base.id and rows_by_arm.get(base.id):
            out["paired"][arm.id] = paired(cfg, rows_by_arm[base.id], rows)
            s, sw = sites_and_sweeps(rows_by_arm[base.id], rows)
            out["sites"][arm.id], out["sweeps"][arm.id] = s, sw
        elif base and arm.id == base.id:
            _, _ = [], {}
    if base and rows_by_arm.get(base.id):  # baseline per-site view (latency, truth accuracy)
        by_site: dict[str, list[dict]] = defaultdict(list)
        for r in rows_by_arm[base.id]:
            for d in r.get("decisions", []):
                by_site[d["site"]].append(d)
        out["sites"][base.id] = [{
            "site": s, "kind": ds[0]["kind"], "n": len(ds), "latency_ms": _mean([d["latency_ms"] for d in ds]),
            "truth_acc": _mean([1.0 if d["correct"] else 0.0 for d in ds if d.get("correct") is not None])
            if any(d.get("correct") is not None for d in ds) else None,
            "parse_fail_rate": _mean([0.0 if d.get("parsed", True) else 1.0 for d in ds]),
        } for s, ds in sorted(by_site.items())]
    return out
