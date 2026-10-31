"""Exploratory (PREREGISTRATION §8): word-level label flips between two segmentations and what predicts them (AMI eval).

    PYTHONPATH=. python -m echobase.explore_flips --out reports/explore_flips.json
A word "flips" for a column when the label of its segment differs between the old and the new segmentation. Reported
per workload and column: the flip rate over all words, by the change kind of the word's new segment
(pilots.segmap.change_map), and by the new/old length ratio of the word's two segments.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilots.ami import meetings
from pilots.segmap import change_map

from .gate import require_g1
from .score_versions import build

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "echobase" / "v2"
COLS = ("action_item", "question", "entity_count", "sentiment", "topic")
PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo")}
RATIO_BINS = (0, 0.5, 0.8, 1.25, 2.0, np.inf)


def ratio_bin(r):
    for lo, hi in zip(RATIO_BINS, RATIO_BINS[1:]):
        if lo <= r < hi:
            return f"[{lo},{hi})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("exploratory eval flips")
    res = {}
    for name, (old_v, new_v) in PAIRS.items():
        tot = defaultdict(lambda: [0, 0])          # col -> [flips, words]
        by_kind = defaultdict(lambda: [0, 0])      # (col, kind)
        by_ratio = defaultdict(lambda: [0, 0])     # (col, ratio bin)
        per_meeting = defaultdict(dict)
        for m in meetings("eval"):
            old, new = build(old_v, m), build(new_v, m)
            kinds = {j: k for k, _, n in change_map(old.segs, new.segs) for j in n}
            lo = [len(s.text.split()) for s in old.segs]
            ln = [len(s.text.split()) for s in new.segs]
            for col in COLS:
                ov = json.loads((V2 / old_v / col / f"{m}.json").read_text())["values"]
                nv = json.loads((V2 / new_v / col / f"{m}.json").read_text())["values"]
                f = n = 0
                for wi, j in new.word_seg.items():
                    i = old.word_seg.get(wi)
                    if i is None:
                        continue
                    flip = ov[i] != nv[j]
                    f += flip; n += 1
                    for key, d in (((col, kinds.get(j, "new")), by_kind), ((col, ratio_bin(ln[j] / lo[i])), by_ratio)):
                        d[key][0] += flip; d[key][1] += 1
                tot[col][0] += f; tot[col][1] += n
                per_meeting[col][m] = f / n
        res[name] = {"flip_rate": {c: f / n for c, (f, n) in tot.items()},
                     "flip_rate_meeting_median": {c: float(np.median(list(d.values()))) for c, d in per_meeting.items()},
                     "by_kind": {f"{c}/{k}": {"rate": f / n, "words": n} for (c, k), (f, n) in sorted(by_kind.items())},
                     "by_length_ratio": {f"{c}/{b}": {"rate": f / n, "words": n} for (c, b), (f, n) in sorted(by_ratio.items())}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, r in res.items():
        print(name, {c: round(v, 3) for c, v in r["flip_rate"].items()})
        print("  kind", {k: round(v["rate"], 3) for k, v in r["by_kind"].items() if k.startswith("action_item")})
        print("  ratio", {k: (round(v["rate"], 3), v["words"]) for k, v in r["by_length_ratio"].items() if k.startswith("action_item")})
