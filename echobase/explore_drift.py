"""Exploratory (AMI dev): segmentation-induced query drift. Same words, two segmentations, same LLM column:
how much do meeting-level aggregate answers change? (word-weighted, so differing segment counts don't bias it)"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pilots.ami import meetings

from .lineage import versions

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "outputs" / "echobase" / "values"


def words_share(segs, vals, pred):
    """Share of the meeting's words that lie in segments whose value satisfies pred (segment-count invariant)."""
    tot = sum(len(s.text.split()) for s in segs)
    return sum(len(s.text.split()) for s, v in zip(segs, vals) if pred(v)) / tot


rows = []
for m in meetings("dev"):
    old, new, _ = versions(m)
    r = {"meeting": m}
    for col, pred in (("action_item", lambda v: v == "yes"), ("question", lambda v: v == "yes"),
                      ("sentiment", lambda v: v == "negative"), ("topic", lambda v: v == "cost")):
        ov = json.loads((V / col / f"{m}.old.json").read_text())
        nv = json.loads((V / col / f"{m}.new.json").read_text())
        r[col] = (words_share(old.segs, ov, pred), words_share(new.segs, nv, pred))
    ov = json.loads((V / "entity_count" / f"{m}.old.json").read_text())
    nv = json.loads((V / "entity_count" / f"{m}.new.json").read_text())
    r["entity_sum"] = (sum(ov), sum(nv))
    # topic distribution shift (total variation distance, word-weighted)
    to = json.loads((V / "topic" / f"{m}.old.json").read_text())
    tn = json.loads((V / "topic" / f"{m}.new.json").read_text())
    def dist(segs, vals):
        c = Counter()
        for s, v in zip(segs, vals):
            c[v] += len(s.text.split())
        t = sum(c.values())
        return {k: x / t for k, x in c.items()}
    do, dn = dist(old.segs, to), dist(new.segs, tn)
    r["topic_tvd"] = 0.5 * sum(abs(do.get(k, 0) - dn.get(k, 0)) for k in set(do) | set(dn))
    r["n_segs"] = (len(old.segs), len(new.segs))
    rows.append(r)

print(f"{'measure':34} {'old (mean)':>10} {'new (mean)':>10} {'median |rel. change|':>22} {'max':>8}")
for col in ("action_item", "question", "sentiment", "topic"):
    o = np.array([r[col][0] for r in rows]); n = np.array([r[col][1] for r in rows])
    rel = np.abs(n - o) / np.maximum(o, 1e-9)
    print(f"{col + ' word-share':34} {o.mean():10.3f} {n.mean():10.3f} {np.median(rel):22.3f} {rel.max():8.3f}")
o = np.array([r["entity_sum"][0] for r in rows], float); n = np.array([r["entity_sum"][1] for r in rows], float)
rel = np.abs(n - o) / np.maximum(o, 1)
print(f"{'SUM(entity_count) per meeting':34} {o.mean():10.1f} {n.mean():10.1f} {np.median(rel):22.3f} {rel.max():8.3f}")
t = np.array([r["topic_tvd"] for r in rows])
print(f"topic distribution TVD per meeting: median {np.median(t):.3f}, max {t.max():.3f}")
print("segments per meeting old/new (mean):", np.mean([r["n_segs"][0] for r in rows]), np.mean([r["n_segs"][1] for r in rows]))
