"""D12 (post-hoc, exploratory; reports/deviations.md): sample-based certification of the aggregate answer. CPU only.

    python -m echobase.explore_aggcert --out reports/explore_aggcert.json

After a re-segmentation, the answer Q(S_new) (word-weighted yes-share of a binary column per meeting) is known exactly
on unchanged segments and on changed cells with identical text (rule `hash`); the remaining cells R are unknown until
scored. A simple random sample of n cells of R (without replacement) estimates their contribution:
  sample:     N_R * mean(w*y)
  difference: sum_R(w*p) + N_R * mean(w*(y - p)), with p the free proxy (rule value, else anchor's old value, else 0).
Normal 95% interval with finite-population correction, reported on Q in percentage points. The cells, rules and anchors
are the replay's (e5_maintenance.pair/load, replay.anchor, rules.derive), from the cached labels of the primary model.
"""
import argparse
import json
import math

import numpy as np

from pilots.ami import meetings
from pilots.segmap import change_map

from .e5_maintenance import V2, pair
from .lineage import provenance
from .llm import COLUMNS
from .replay import anchor
from .rules import derive

COLS = ("action_item", "question")
FRACS = (0.05, 0.10, 0.20, 0.40)
MIN_N, RESAMPLES, Z = 10, 1000, 1.959963984540054
WORKLOADS = ("w1", "w3")


def units(workload):
    """Per meeting and column: total words, exact known yes-words, and the unknown cells R as (w, y, p)."""
    out = []
    for m in meetings("eval"):
        old, new, on, nn = pair(workload, m)
        prov = provenance(old, new)
        kinds = {j: k for k, _, n in change_map(old.segs, new.segs) for j in n}
        n_changed = sum(1 for j in range(len(new.segs)) if kinds.get(j, "new") != "unchanged")
        for col in COLS:
            od = json.loads((V2 / on / col / f"{m}.json").read_text())
            nd = json.loads((V2 / nn / col / f"{m}.json").read_text())
            assert len(nd["values"]) == len(new.segs) and len(od["values"]) == len(old.segs)
            lens, yes = nd["lens"], [v == "yes" for v in nd["values"]]
            known, R = 0, []
            for j, seg in enumerate(new.segs):
                w, y = lens[j], yes[j]
                if kinds.get(j, "new") == "unchanged":
                    known += w * y
                    continue
                v, rule = derive(COLUMNS[col].kind, seg.text, prov[j], old.segs, od["values"])
                if rule == "hash":
                    known += w * y
                    continue
                a = anchor(seg, prov[j], old.segs)
                p = (v == "yes") if rule is not None else (od["values"][a] == "yes") if a is not None else False
                R.append((w, float(y), float(p)))
            out.append({"meeting": m, "col": col, "W": sum(lens), "known": known, "R": np.array(R).reshape(-1, 3),
                        "changed": n_changed})
    return out


def simulate(u, frac, rng):
    """Half-widths (points) and coverage flags of both estimators over RESAMPLES draws; calls per draw."""
    R, W = u["R"], u["W"]
    N = len(R)
    truth = (u["known"] + (R[:, 0] * R[:, 1]).sum()) / W
    if N == 0:
        return {"calls": 0, "sample": ([0.0] * RESAMPLES, [True] * RESAMPLES),
                "difference": ([0.0] * RESAMPLES, [True] * RESAMPLES)}
    n = min(N, max(MIN_N, math.ceil(frac * N)))
    fpc = (1 - n / N) if N > 1 else 0.0
    wy, wd = R[:, 0] * R[:, 1], R[:, 0] * (R[:, 1] - R[:, 2])
    base = (R[:, 0] * R[:, 2]).sum()
    res = {"calls": n, "sample": ([], []), "difference": ([], [])}
    for _ in range(RESAMPLES):
        idx = rng.choice(N, n, replace=False)
        for name, x, off in (("sample", wy, 0.0), ("difference", wd, base)):
            s = x[idx]
            est = (u["known"] + off + N * s.mean()) / W
            se = N * math.sqrt(fpc * (s.var(ddof=1) if n > 1 else 0.0) / n) / W
            hw = Z * se
            res[name][0].append(100 * hw)
            res[name][1].append(abs(est - truth) <= hw + 1e-12)
    return res


def run():
    rng = np.random.default_rng(0)
    out = {"design": "reports/deviations.md D12", "fractions": FRACS, "resamples": RESAMPLES, "min_n": MIN_N,
           "workloads": {}}
    for wl in WORKLOADS:
        us = units(wl)
        changed = sum(u["changed"] for u in us) // len(COLS)          # changed segments (= cells per column)
        unknown = {c: sum(len(u["R"]) for u in us if u["col"] == c) for c in COLS}
        wres = {"changed_segments": changed, "unknown_cells": unknown, "by": {}}
        for col in COLS:
            cu = [u for u in us if u["col"] == col]
            for f in FRACS:
                sims = [simulate(u, f, rng) for u in cu]
                calls = sum(s["calls"] for s in sims)
                row = {"calls": calls, "call_share": calls / changed}
                for est in ("sample", "difference"):
                    hw = np.concatenate([s[est][0] for s in sims])
                    cov = np.concatenate([s[est][1] for s in sims])
                    per_m = [float(np.median(s[est][0])) for s in sims]
                    row[est] = {"median_halfwidth_pts": float(np.median(hw)), "p90_halfwidth_pts":
                                float(np.percentile(hw, 90)), "max_meeting_median_halfwidth_pts": max(per_m),
                                "coverage": float(cov.mean()),
                                "min_meeting_coverage": min(float(np.mean(s[est][1])) for s in sims)}
                wres["by"][f"{col}@{f}"] = row
        out["workloads"][wl] = wres
    # rules fixed in D12
    certifies = {}
    for est in ("sample", "difference"):
        for f in FRACS:
            covs = [out["workloads"][wl]["by"][f"{c}@{f}"][est]["coverage"] for wl in WORKLOADS for c in COLS]
            certifies[f"{est}@{f}"] = min(covs) >= 0.93
    helps = {f: all(out["workloads"][wl]["by"][f"{c}@{f}"]["difference"]["median_halfwidth_pts"] <=
                    0.8 * out["workloads"][wl]["by"][f"{c}@{f}"]["sample"]["median_halfwidth_pts"]
                    for wl in WORKLOADS for c in COLS) for f in FRACS}
    head = {}
    for wl in WORKLOADS:
        for c in COLS:
            for est in ("difference", "sample"):
                ok = [f for f in FRACS if certifies[f"{est}@{f}"] and
                      out["workloads"][wl]["by"][f"{c}@{f}"][est]["median_halfwidth_pts"] <= 2.0]
                head[f"{wl}/{c}/{est}"] = ({"fraction": min(ok), "call_share":
                                            out["workloads"][wl]["by"][f"{c}@{min(ok)}"]["call_share"]}
                                           if ok else None)
    out["decision"] = {"certifies_95": certifies, "proxy_helps": {str(k): v for k, v in helps.items()},
                       "smallest_fraction_2pts": head}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["decision"], indent=1))
