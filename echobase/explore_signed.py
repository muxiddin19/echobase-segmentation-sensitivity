"""Post-hoc exploratory (deviation D5): direction and absolute size of drift, per workload and share query.

    python -m echobase.explore_signed --out reports/explore_signed.json [--model Qwen/Qwen2.5-0.5B-Instruct]
Per unit: old share, new share, absolute change (new - old, percentage points) and signed relative change
((new - old) / old, same 0.02 guard as D). Reported: medians with 95% intervals (power_sim.median_ci), and how many
units move up. Share queries only (Q1-Q4, Qd1), so that a direction is defined.
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
PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo"), "W3": ("w3old", "w1old"),
         "E6_k2": ("manual", "merge2"), "E6_k4": ("manual", "merge4"), "E6_k8": ("manual", "merge8")}


def summarize(old, new):
    old, new = np.array(old), np.array(new)
    absd = new - old
    keep = old >= drift.GUARD
    rel = absd[keep] / old[keep]
    out = {"units": len(old), "valid": int(keep.sum()), "old_share_median": float(np.median(old)),
           "up": int((absd > 0).sum()), "down": int((absd < 0).sum())}
    for name, x in (("abs", absd), ("absmag", np.abs(absd)), ("rel", rel)):  # absmag: size regardless of direction
        if len(x) >= 3:
            lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
            out[name] = {"median": float(np.median(x)), "ci": [float(lo), float(hi)], "ci_method": how}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--cache", help="cache dir under outputs/echobase (D6 prompt variants), overrides --model's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc signed drift")
    drift.V2 = ROOT / "outputs" / "echobase" / a.cache if a.cache else cache_root(ROOT, a.model)
    res = {"model": a.model, "ami": {}, "wiki": {}}
    for name, (o, n) in PAIRS.items():
        if not all((drift.V2 / v / "action_item" / f"{m}.json").exists() for v in (o, n) for m in meetings("eval")):
            continue
        res["ami"][name] = {}
        for q, (col, target) in drift.AMI_QUERIES.items():
            olds, news = [], []
            for m in meetings("eval"):
                vo, lo = drift.ami_load(o, col, m)
                vn, ln = drift.ami_load(n, col, m)
                olds.append(drift.share(vo, lo, target)); news.append(drift.share(vn, ln, target))
            res["ami"][name][q] = summarize(olds, news)
    from .wiki import articles
    if (drift.V2 / "wiki" / "para" / "doc_date").exists():
        olds, news = [], []
        for art in articles("test"):
            def load(ch):
                d = json.loads((drift.V2 / "wiki" / ch / "doc_date" / f"{art['id']}.json").read_text())
                return d["values"], [e - s for s, e in d["spans"]]
            olds.append(drift.share(*load("fixed"), "yes")); news.append(drift.share(*load("para"), "yes"))
        res["wiki"]["Qd1"] = summarize(olds, news)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, qs in res["ami"].items():
        for q, r in qs.items():
            print(f"{name:6} {q}: old {r['old_share_median']:.3f} up {r['up']}/{r['units']} "
                  f"abs {r['abs']['median']:+.4f} {r['abs']['ci']} rel {r.get('rel', {}).get('median', float('nan')):+.3f}")
    for q, r in res["wiki"].items():
        print(f"wiki   {q}: old {r['old_share_median']:.3f} up {r['up']}/{r['units']} abs {r['abs']['median']:+.4f} "
              f"{r['abs']['ci']} rel {r['rel']['median']:+.3f} {r['rel']['ci']}")
