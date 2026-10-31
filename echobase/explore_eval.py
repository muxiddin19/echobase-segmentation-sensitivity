"""Exploratory drift on AMI eval / Wikipedia test (PREREGISTRATION §8; no decision rules). CPU only.

    python -m echobase.explore_eval --out reports/explore_eval.json                                  # primary model
    python -m echobase.explore_eval --model Qwen/Qwen2.5-0.5B-Instruct --out reports/explore_eval_model2.json   # E4
Per workload pair and query: valid units, guard-excluded, median D and its 95% CI (power_sim.median_ci, the
confirmatory interval). Pairs: W1 w1old->manual, W2 manual->placebo, W3 w3old->w1old, E6 manual->merge{2,4,8};
Wikipedia fixed->para. Also the paired per-meeting difference D(W2) - D(W1) for Q1. A pair whose caches are
incomplete is skipped and listed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pilots.ami import meetings

from . import drift
from .gate import require_g1
from .llm import PRIMARY, cache_root
from .power_sim import median_ci

ROOT = Path(__file__).resolve().parents[1]
COLS = ("action_item", "question", "entity_count", "sentiment", "topic")
PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo"), "W3": ("w3old", "w1old"),
         "E6_k2": ("manual", "merge2"), "E6_k4": ("manual", "merge4"), "E6_k8": ("manual", "merge8")}


def stat(d):
    x = np.array([v for v in d.values() if v is not None], float)
    rec = {"units": len(d), "excluded_guard": len(d) - len(x), "valid": len(x)}
    if len(x) < 3:
        return rec
    lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
    return dict(rec, median=float(np.median(x)), ci=[float(lo), float(hi)], ci_method=how)


def complete(old, new):
    return all((drift.V2 / v / c / f"{m}.json").exists() for v in (old, new) for c in COLS for m in meetings("eval"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--cache", help="cache dir under outputs/echobase (D6 prompt variants), overrides --model's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("exploratory eval analysis")
    drift.V2 = ROOT / "outputs" / "echobase" / a.cache if a.cache else cache_root(ROOT, a.model)  # ami_load / wiki_drift read this module global at call time
    res = {"model": a.model, "ami": {}, "skipped": [], "wiki": {}}
    per_q1 = {}
    for name, (old, new) in PAIRS.items():
        if not complete(old, new):
            res["skipped"].append(name)
            continue
        dr = drift.ami_drift(old, new, "eval")
        res["ami"][name] = {q: dict(stat(d), per_unit=d) for q, d in dr.items()}
        per_q1[name] = dr["Q1"]
    if "W1" in per_q1 and "W2" in per_q1:
        diff = {m: per_q1["W2"][m] - per_q1["W1"][m] for m in per_q1["W1"]
                if per_q1["W1"][m] is not None and per_q1["W2"][m] is not None}
        x = np.array(list(diff.values()))
        lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
        res["W2_minus_W1_Q1"] = {"n": len(x), "median": float(np.median(x)), "ci": [float(lo), float(hi)],
                                 "ci_method": how, "w2_larger": int((x > 0).sum())}
    if (drift.V2 / "wiki" / "para" / "doc_topic").exists():
        wd = drift.wiki_drift("test")
        res["wiki"] = {q: dict(stat(d), per_unit=d) for q, d in wd.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, qs in res["ami"].items():
        print(name, "  ".join(f"{q}={r.get('median', float('nan')):.3f}[{r.get('ci', [np.nan, np.nan])[0]:.3f},"
                              f"{r.get('ci', [np.nan, np.nan])[1]:.3f}]({r['valid']})" for q, r in qs.items()))
    for q, r in res["wiki"].items():
        print("wiki", q, f"{r.get('median', float('nan')):.3f}", r.get("ci"), r["valid"])
    print("W2-W1", res.get("W2_minus_W1_Q1"), "skipped", res["skipped"])
