"""Post-hoc scale analysis (deviation D5): drift and cost on the AMI meetings outside dev/eval, second model.

    PYTHONPATH=. python -m echobase.explore_scale --meetings logs/scale_done.txt --out reports/explore_scale.json
For W1 (w1old -> manual) and W2 (manual -> placebo) with Qwen2.5-0.5B: median D of Q1/Q2 and its interval over all
scale meetings; the same on nested random subsets of 16, 32, 64 meetings and all (seed 0) to show how the interval
narrows with corpus size; signed absolute change; and the LLM calls per version (the cost of materializing and of
recomputing a view), which grow linearly with the corpus.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import drift
from .gate import require_g1
from .llm import cache_root
from .power_sim import median_ci

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo")}


def stat(x):
    x = np.array([v for v in x if v is not None], float)
    lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
    return {"valid": len(x), "median": float(np.median(x)), "ci": [float(lo), float(hi)], "ci_method": how}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cache", help="D7: cache dir under outputs/echobase (the 1.5B scale run), overrides --model's")
    a = ap.parse_args()
    require_g1("post-hoc scale analysis")
    drift.V2 = ROOT / "outputs" / "echobase" / a.cache if a.cache else cache_root(ROOT, a.model)
    ms = [l.strip() for l in Path(a.meetings).read_text().splitlines() if l.strip()]
    cols = ("action_item", "question", "entity_count", "sentiment", "topic")
    ms = [m for m in ms if all((drift.V2 / v / c / f"{m}.json").exists()
                               for v in ("w1old", "manual", "placebo") for c in cols)]
    order = list(ms)
    np.random.default_rng(0).shuffle(order)
    res = {"model": a.model, "meetings": len(ms), "pairs": {}, "calls": {}}
    q1 = {}
    for name, (o, n) in PAIRS.items():
        d = drift.ami_drift(o, n, "eval", only=ms)
        q1[name] = d["Q1"]
        res["pairs"][name] = {}
        for q in ("Q1", "Q2"):
            full = stat(d[q].values())
            nested = {}
            for k in (16, 32, 64, len(order)):
                nested[str(k)] = stat([d[q][m] for m in order[:k]])
            absd = []
            col, target = drift.AMI_QUERIES[q]
            for m in ms:
                vo, lo = drift.ami_load(o, col, m)
                vn, ln = drift.ami_load(n, col, m)
                absd.append(drift.share(vn, ln, target) - drift.share(vo, lo, target))
            absd = np.array(absd)
            res["pairs"][name][q] = dict(full, nested=nested, abs_median=float(np.median(absd)),
                                         up=int((absd > 0).sum()), down=int((absd < 0).sum()))
    diff = np.array([q1["W2"][m] - q1["W1"][m] for m in ms if q1["W1"][m] is not None and q1["W2"][m] is not None])
    lo, hi, how = median_ci(diff, np.random.default_rng(0), 10_000)
    res["W2_minus_W1_Q1"] = {"n": len(diff), "median": float(np.median(diff)), "ci": [float(lo), float(hi)],
                             "ci_method": how, "w2_larger": int((diff > 0).sum())}
    print("W2-W1 Q1", res["W2_minus_W1_Q1"])
    for v in ("manual", "w1old", "placebo"):
        res["calls"][v] = sum(len(json.loads((drift.V2 / v / "action_item" / f"{m}.json").read_text())["values"])
                              for m in ms) * 5  # five columns
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("meetings", len(ms), "calls", res["calls"])
    for name, qs in res["pairs"].items():
        for q, r in qs.items():
            print(name, q, f"median {r['median']:.3f} {r['ci']} valid {r['valid']} abs {r['abs_median']:+.4f} "
                  f"up {r['up']} down {r['down']}", {k: (round(v['median'], 3), [round(c, 3) for c in v['ci']])
                                                     for k, v in r['nested'].items()})
