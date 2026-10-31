"""Replay maintenance policies on cached values (CPU only, AMI dev): LLM calls vs error vs full recompute.

    python -m echobase.replay --columns action_item,question,entity_count,sentiment,topic --trials 50

Every old/new cell value is cached by echobase.measure_decomp, so the 'truth' of a recompute is known and the
number of LLM calls a policy would make is counted exactly. Only changed cells are maintained (unchanged
cells are reused by every policy).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilots.ami import meetings
from pilots.segmap import change_map

from .certify import audit
from .lineage import provenance, versions
from .llm import COLUMNS
from .rules import derive

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "echobase" / "values"


def anchor(seg, prov, old_segs):
    """Most-overlapping contributing old segment (same speaker first), or None."""
    best, best_ov = None, 0.0
    for i in prov["contrib"]:
        o = old_segs[i]
        ov = max(0.0, min(o.end, seg.end) - max(o.start, seg.start)) + (1e6 if o.speaker == seg.speaker else 0)
        if ov > best_ov:
            best, best_ov = i, ov
    return best


MARGINS = ROOT / "outputs" / "echobase" / "margins"
CONF_EDGES = (2.0, 5.0)  # logit-margin bins: low < 2 <= medium < 5 <= high (fixed before e03 results)


def conf_bin(margin):
    if margin is None:
        return "na"
    return "low" if margin < CONF_EDGES[0] else "mid" if margin < CONF_EDGES[1] else "high"


def cells(columns, split="dev"):
    """One record per changed cell: column, change kind, rule value/name, anchor value, truth, confidence."""
    out = []
    for m in meetings(split):
        old, new, _ = versions(m)
        prov = provenance(old, new)
        kinds = {j: k for k, _, n in change_map(old.segs, new.segs) for j in n}
        for col in columns:
            ov = json.loads((CACHE / col / f"{m}.old.json").read_text())
            nv = json.loads((CACHE / col / f"{m}.new.json").read_text())
            mf = MARGINS / col / f"{m}.old.json"
            om = json.loads(mf.read_text()) if mf.exists() else None
            kind = COLUMNS[col].kind
            for j, seg in enumerate(new.segs):
                k = kinds.get(j, "new")
                if k == "unchanged":
                    continue
                v, rule = derive(kind, seg.text, prov[j], old.segs, ov)
                a = anchor(seg, prov[j], old.segs)
                contrib = prov[j]["contrib"]
                out.append({"meeting": m, "col": col, "kind": kind, "change": k, "rule": rule, "rule_val": v,
                            "anchor_val": ov[a] if a is not None else None, "truth": nv[j],
                            # weakest contributing old cell (rules) / the anchor's own margin (anchor reuse)
                            "rule_conf": min(om[i] for i in contrib) if om and contrib else None,
                            "anchor_conf": om[a] if om and a is not None else None})
    return out


def paired(cs):
    """Rule vs anchor reuse on identical cells: only cells where a non-hash rule fires AND an anchor exists.
    {(col, rule): {"n", "rule_err", "anchor_err", "pairs"}}, pairs = [(rule_val, anchor_val, truth)]."""
    by = defaultdict(list)
    for c in cs:
        if c["rule"] in (None, "hash") or c["anchor_val"] is None:
            continue
        by[(c["col"], c["rule"])].append((c["rule_val"], c["anchor_val"], c["truth"]))
    return {k: {"n": len(p), "rule_err": sum(r != t for r, _, t in p) / len(p),
                "anchor_err": sum(a != t for _, a, t in p) / len(p), "pairs": p} for k, p in by.items()}


def run_policy(policy, cs, rng, eps, delta):
    """Returns (llm_calls, wrong_cells, n_cells, strata_info)."""
    if policy == "full":
        return len(cs), 0, len(cs), {}
    if policy == "hash":
        calls = sum(c["rule"] != "hash" for c in cs)
        return calls, 0, len(cs), {}
    strata = defaultdict(list)
    calls = 0
    for idx, c in enumerate(cs):
        if policy == "anchor_cert":   # sIVM-style: provisional = anchor value, strata by column
            if c["rule"] == "hash":
                continue              # identical text: reuse exactly, like the hash baseline
            prov_val, key = c["anchor_val"], (c["col"],)
        elif policy == "echobase":    # rules propose where they apply, anchors elsewhere; strata by provenance
            if c["rule"] == "hash":
                continue
            if c["rule"] is not None:
                prov_val, key = c["rule_val"], (c["col"], "rule", c["rule"])
            else:
                prov_val, key = c["anchor_val"], (c["col"], "anchor", c["change"])
        elif policy == "echobase_conf":  # e03: as echobase, strata further split by upstream confidence
            if c["rule"] == "hash":
                continue
            if c["rule"] is not None:
                prov_val, key = c["rule_val"], (c["col"], "rule", c["rule"], conf_bin(c["rule_conf"]))
            else:
                prov_val, key = c["anchor_val"], (c["col"], "anchor", c["change"], conf_bin(c["anchor_conf"]))
        else:
            raise ValueError(policy)
        if prov_val is None:          # no anchor: must recompute
            calls += 1
            continue
        strata[key].append((idx, prov_val))
    wrong, info = 0, {}
    alpha = delta / max(len(strata), 1)
    for key, items in strata.items():
        prov = [v for _, v in items]
        truth = [cs[i]["truth"] for i, _ in items]
        final, n_calls, ok, dh = audit(prov, truth, rng, eps, alpha)
        calls += n_calls
        w = sum(f != t for f, t in zip(final, truth))
        wrong += w
        info["/".join(map(str, key))] = {"N": len(items), "calls": n_calls, "certified": ok,
                                          "true_err": w / len(items), "violation": ok and w / len(items) > eps}
    return calls, wrong, len(cs), info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", default="action_item,question,entity_count,sentiment,topic")
    ap.add_argument("--policies", default="full,hash,anchor_cert,echobase")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--out", default="outputs/echobase/replay_dev.json")
    a = ap.parse_args()
    cs = cells(a.columns.split(","))
    res = {}
    for pol in a.policies.split(","):
        runs = [run_policy(pol, cs, np.random.default_rng([t, 7]), a.eps, a.delta) for t in range(a.trials)]
        calls = [r[0] for r in runs]
        err = [r[1] / r[2] for r in runs]
        viol = [any(s["violation"] for s in r[3].values()) for r in runs]
        res[pol] = {"cells": runs[0][2], "calls_mean": float(np.mean(calls)), "call_share": float(np.mean(calls)) / runs[0][2],
                    "cell_error_mean": float(np.mean(err)), "cell_error_max": float(np.max(err)),
                    "any_stratum_violation_rate": float(np.mean(viol)), "strata_trial0": runs[0][3]}
        print(f"{pol:12} call_share={res[pol]['call_share']:.3f} err_mean={res[pol]['cell_error_mean']:.4f} "
              f"err_max={res[pol]['cell_error_max']:.4f} viol={res[pol]['any_stratum_violation_rate']:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"columns": a.columns, "eps": a.eps, "delta": a.delta, "trials": a.trials,
                                       "policies": res}, indent=1))
