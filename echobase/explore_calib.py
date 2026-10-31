"""Post-hoc calibration baselines (deviation D6): drift caused by other routine changes, on a FIXED segmentation.

    PYTHONPATH=. python -m echobase.explore_calib --out reports/explore_calib.json
D between two caches of the same version (AMI eval, manual segments): a model swap (1.5B vs 0.5B, 3B, Phi-3.5-mini)
and a prompt paraphrase (1.5B and 0.5B, frozen prompt vs reworded question). Same statistic, guard and interval as the
segmentation workloads, so the numbers can be read side by side. Pairs whose caches are incomplete are skipped.
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
PAIRS = {  # name: (cache A, cache B); both on the manual segmentation
    "model 1.5B->0.5B": ("v2", "v2-Qwen2.5-0.5B-Instruct"),
    "model 1.5B->3B": ("v2", "v2-Qwen2.5-3B-Instruct"),
    "model 1.5B->Phi-3.5-mini": ("v2", "v2-Phi-3.5-mini-instruct"),
    "model 1.5B->7B": ("v2", "v2-Qwen2.5-7B-Instruct"),
    "prompt 1.5B paraphrase": ("v2", "v2-para"),
    "prompt 1.5B paraphrase 2": ("v2", "v2-para2"),  # D7: three further rewordings
    "prompt 1.5B paraphrase 3": ("v2", "v2-para3"),
    "prompt 1.5B paraphrase 4": ("v2", "v2-para4"),
    "prompt 0.5B paraphrase": ("v2-Qwen2.5-0.5B-Instruct", "v2-Qwen2.5-0.5B-Instruct-para"),
    "context 1.5B (ctx, manual)": ("v2", "v2-ctx"),
}


def load(cache, col, m):
    d = json.loads((EB / cache / "manual" / col / f"{m}.json").read_text())
    return d["values"], d["lens"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc calibration")
    ms = meetings("eval")
    res = {"skipped": [], "pairs": {}}
    for name, (ca, cb) in PAIRS.items():
        if not all((EB / c / "manual" / col / f"{m}.json").exists()
                   for c in (ca, cb) for col, _ in drift.AMI_QUERIES.values() for m in ms):
            res["skipped"].append(name)
            continue
        res["pairs"][name] = {}
        for q, (col, target) in drift.AMI_QUERIES.items():
            d, absd = [], []
            for m in ms:
                qa, qb = drift.share(*load(ca, col, m), target), drift.share(*load(cb, col, m), target)
                d.append(drift.rel(qa, qb)); absd.append(qb - qa)
            x = np.array([v for v in d if v is not None], float)
            r = {"valid": len(x), "abs_median": float(np.median(absd))}
            if len(x) >= 3:
                lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
                r.update(median=float(np.median(x)), ci=[float(lo), float(hi)], ci_method=how)
            res["pairs"][name][q] = r
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, qs in res["pairs"].items():
        print(name, {q: (round(r.get("median", float("nan")), 3), r["valid"], round(100 * r["abs_median"], 1))
                     for q, r in qs.items()})
    print("skipped", res["skipped"])
