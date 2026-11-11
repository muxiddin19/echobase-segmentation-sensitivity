"""Post-hoc exploratory (deviation D5): how close each segmentation's answer is to human ground truth (AMI eval).

    python -m echobase.explore_gold --out reports/explore_gold.json [--model Qwen/Qwen2.5-0.5B-Instruct]
Gold word labels come from the AMI manual dialogue acts, with the type sets fixed in Pilot B before this study
(pilots/run_pilotB_labels.py): question <- elicit acts {el.inf, el.ass}; action item <- suggestion acts {sug}, a
proxy (AMI has no action-item act). A reference word is gold-positive when it lies (by its time midpoint) inside a
positive act of the same speaker. Per meeting and segmentation version: the gold share, the predicted share
(word-weighted, as Q1/Q2), the absolute error, and word-level precision / recall / F1.
"""
import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilots.ami import dialogue_acts, meetings

from .gate import require_g1
from .lineage import system_version, versions
from .llm import PRIMARY, cache_root
from .score_versions import W3_OLD, build

ROOT = Path(__file__).resolve().parents[1]
GOLD = {"question": {"el.inf", "el.ass"}, "action_item": {"sug"}}
VERSIONS = ("manual", "w1old", "placebo", "w3old", "merge2", "merge4", "merge8")


def gold_words(m, words, types):
    """Boolean per reference word: inside a dialogue act of the given types, same speaker."""
    by_spk = defaultdict(list)
    for da in dialogue_acts(m):
        if da["type"] in types:
            by_spk[da["speaker"]].append((da["start"], da["end"]))
    for v in by_spk.values():
        v.sort()
    out = []
    for w in words:
        spans, mid = by_spk.get(w["speaker"], []), (w["start"] + w["end"]) / 2
        i = bisect.bisect_right(spans, (mid, float("inf"))) - 1
        out.append(i >= 0 and spans[i][0] <= mid <= spans[i][1])
    return out


def version(v, m):
    if v == "w3old":
        return system_version(m, W3_OLD / f"{m}.json", "w3o")[0]
    return build(v, m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PRIMARY)
    ap.add_argument("--cache", help="cache dir under outputs/echobase (D6 prompt variants), overrides --model's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc gold comparison")
    V = ROOT / "outputs" / "echobase" / a.cache if a.cache else cache_root(ROOT, a.model)
    res = {"model": a.model, "gold_types": {c: sorted(t) for c, t in GOLD.items()}, "per_meeting": {}, "summary": {}}
    for m in meetings("eval"):
        words = versions(m)[2]
        res["per_meeting"][m] = {}
        for col, types in GOLD.items():
            g = np.array(gold_words(m, words, types))
            rec = {"gold_share": float(g.mean())}
            for v in VERSIONS:
                f = V / v / col / f"{m}.json"
                if not f.exists():
                    continue
                ver, vals = version(v, m), json.loads(f.read_text())["values"]
                pred = np.array([wi in ver.word_seg and vals[ver.word_seg[wi]] == "yes" for wi in range(len(words))])
                tp = int((pred & g).sum())
                p = tp / pred.sum() if pred.sum() else 0.0
                r = tp / g.sum() if g.sum() else 0.0
                rec[v] = {"share": float(pred.mean()), "abs_err": float(abs(pred.mean() - g.mean())),
                          "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
            res["per_meeting"][m][col] = rec
    for col in GOLD:
        res["summary"][col] = {"gold_share_median": float(np.median([res["per_meeting"][m][col]["gold_share"]
                                                                     for m in res["per_meeting"]]))}
        for v in VERSIONS:
            recs = [res["per_meeting"][m][col][v] for m in res["per_meeting"] if v in res["per_meeting"][m][col]]
            if recs:
                res["summary"][col][v] = {k: float(np.median([r[k] for r in recs])) for k in recs[0]}
                res["summary"][col][v]["meetings"] = len(recs)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for col, s in res["summary"].items():
        print(col, "gold share", round(s["gold_share_median"], 3))
        for v in VERSIONS:
            if v in s:
                print(f"  {v:8} share {s[v]['share']:.3f} abs_err {s[v]['abs_err']:.3f} P {s[v]['precision']:.3f} "
                      f"R {s[v]['recall']:.3f} F1 {s[v]['f1']:.3f}")
