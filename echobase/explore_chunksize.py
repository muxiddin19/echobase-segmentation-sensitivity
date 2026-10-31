"""Post-hoc exploratory (deviation D7): a pipeline-style chunker revision, a change of window size.

    PYTHONPATH=. python -m echobase.explore_chunksize --out reports/explore_chunksize.json
The locked fixed chunker cuts Wikipedia test articles into 120-word windows; post_score wikisize re-cuts them into 100-
and 150-word windows with the primary model. Reported for 120 -> 100 and 120 -> 150: D of the dated share (Qd1, the
H4 query, same 0.02 guard), its 95% interval, and the signed absolute change; next to it H4's fixed -> paragraph D.
Also the means and the number of articles whose dated share changes at all (D is lattice-valued at these sizes).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import drift
from .gate import require_g1
from .llm import PRIMARY, cache_root
from .power_sim import median_ci
from .wiki import articles

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {"120->100": ("fixed", "fixed100"), "120->150": ("fixed", "fixed150"), "fixed->para (H4)": ("fixed", "para")}


def dated_share(art, chunker):
    d = json.loads((cache_root(ROOT, PRIMARY) / "wiki" / chunker / "doc_date" / f"{art}.json").read_text())
    return drift.share(d["values"], [e - s for s, e in d["spans"]], "yes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc chunk-size drift")
    ids = [art["id"] for art in articles("test")]
    res = {}
    for name, (o, n) in PAIRS.items():
        old = [dated_share(i, o) for i in ids]
        new = [dated_share(i, n) for i in ids]
        d = np.array([v for v in (drift.rel(x, y) for x, y in zip(old, new)) if v is not None], float)
        absd = 100 * (np.array(new) - np.array(old))
        lo, hi, how = median_ci(d, np.random.default_rng(0), 10_000)
        alo, ahi, _ = median_ci(absd, np.random.default_rng(0), 10_000)
        res[name] = {"valid": len(d), "median": float(np.median(d)), "ci": [float(lo), float(hi)], "ci_method": how,
                     "abs_median": float(np.median(absd)), "abs_ci": [float(alo), float(ahi)],
                     "up": int((absd > 0).sum()), "units": len(ids),
                     "mean": float(np.mean(d)), "abs_mean": float(np.mean(absd)), "changed": int((absd != 0).sum())}
        r = res[name]
        print(f"{name:17} D {r['median']:.3f} {r['ci']} abs {r['abs_median']:+.2f} pts {r['abs_ci']} "
              f"up {r['up']}/{r['units']}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
