"""Post-hoc exploratory (deviation D7): drift restricted to confident labels.

    PYTHONPATH=. python -m echobase.explore_confident --out reports/explore_confident.json
A weak scorer could drift only because many of its labels are near-ties. For W1 (w1old -> manual) and W2 (manual ->
placebo) and the share queries Q1 (action items) and Q2 (questions), the word-weighted share is recomputed over the
segments whose first-token logit margin (top-1 minus top-2 answer logit) is at least MARGIN on each side, and D is
compared with the unrestricted D on the same meetings. MARGIN was fixed in the deviation log before any run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pilots.ami import meetings

from . import drift
from .gate import require_g1
from .power_sim import median_ci

ROOT = Path(__file__).resolve().parents[1]
EB = ROOT / "outputs" / "echobase"
MARGIN = 2.0
CACHES = {"Qwen2.5-0.5B": "v2-Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B": "v2", "Qwen2.5-3B": "v2-Qwen2.5-3B-Instruct",
          "Qwen2.5-7B": "v2-Qwen2.5-7B-Instruct", "Phi-3.5-mini": "v2-Phi-3.5-mini-instruct"}
PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo")}
QUERIES = {"Q1": ("action_item", "yes"), "Q2": ("question", "yes")}


def load(cache, version, col, m):
    d = json.loads((EB / cache / version / col / f"{m}.json").read_text())
    return d["values"], d["margins"], d["lens"]


def shares(cache, version, col, m, target):
    """(share over all segments, share over confident segments, fraction of words that are confident)."""
    v, mg, ln = load(cache, version, col, m)
    keep = [i for i in range(len(v)) if mg[i] >= MARGIN]
    conf = drift.share([v[i] for i in keep], [ln[i] for i in keep], target) if keep else None
    return drift.share(v, ln, target), conf, sum(ln[i] for i in keep) / max(sum(ln), 1)


def stat(x):
    x = np.array([v for v in x if v is not None], float)
    if len(x) < 3:
        return {"valid": len(x)}
    lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
    return {"valid": len(x), "median": float(np.median(x)), "ci": [float(lo), float(hi)], "ci_method": how}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc confident-label drift")
    res = {"margin": MARGIN, "models": {}}
    for label, cache in CACHES.items():
        if not (EB / cache / "placebo").exists():
            continue
        res["models"][label] = {}
        for w, (o, n) in PAIRS.items():
            for q, (col, target) in QUERIES.items():
                d_all, d_conf, kept = [], [], []
                for m in meetings("eval"):
                    ao, co, ko = shares(cache, o, col, m, target)
                    an, cn, kn = shares(cache, n, col, m, target)
                    d_all.append(drift.rel(ao, an))
                    d_conf.append(drift.rel(co, cn) if co is not None and cn is not None else None)
                    kept.append((ko + kn) / 2)
                r = {"all": stat(d_all), "confident": stat(d_conf), "kept_words_median": float(np.median(kept))}
                res["models"][label][f"{w}/{q}"] = r
                print(f"{label:13} {w} {q}: D all {r['all'].get('median', float('nan')):.3f} "
                      f"confident {r['confident'].get('median', float('nan')):.3f} {r['confident'].get('ci')} "
                      f"(n={r['confident']['valid']}, words kept {r['kept_words_median']:.2f})", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
