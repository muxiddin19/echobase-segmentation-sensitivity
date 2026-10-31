"""E5 maintenance consequences on AMI eval (exploratory; plan v2.2 §6 E5, PREREGISTRATION §8). CPU only.

    python -m echobase.e5_maintenance --workload w1 --out reports/e5_w1.json
    python -m echobase.e5_maintenance --workload w3 --out reports/e5_w3.json

Cells come from the v2 eval caches (E2) instead of the dev-only caches that the locked replay.cells reads; the record
format, the policies (replay.run_policy) and the audit (certify.audit) are the locked code, unchanged.
- workload w1: old = pyannote 3.1 known count (w1old), new = AMI manual (H1's pair);
- workload w3: old = pyannote 3.1 estimated count (w3old), new = known count (w1old).
Outputs, per policy and eps: call share and cell error over trials, plus a meeting bootstrap that re-runs the whole
policy in each resample with a fresh trial seed (plan: "the policy is re-run inside each bootstrap resample"); kappa of
each structural rule vs the recomputed truth next to the majority-class error; tokens per policy (calls x mean tokens of
a changed cell, primary tokenizer); adversarial validity (strata injected to true error eps + 0.02).
"""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pilots.ami import meetings
from pilots.segmap import change_map

from .gate import require_g1
from .lineage import provenance, system_version, versions
from .llm import COLUMNS, PRIMARY, cache_root
from .replay import anchor, conf_bin, run_policy
from .rules import derive
from .certify import audit
from .score_versions import W3_OLD

ROOT = Path(__file__).resolve().parents[1]
V2 = cache_root(ROOT, PRIMARY)
COLS = ("action_item", "question", "entity_count", "sentiment", "topic")
POLICIES = ("full", "hash", "anchor_cert", "echobase", "echobase_conf")
EPS = (0.02, 0.05, 0.10, 0.15)
DELTA, TRIALS, BOOT = 0.05, 100, 200


def pair(workload, m):
    """(old version, new version, old cache name, new cache name) over the same reference words."""
    w1old, manual, _ = versions(m)
    if workload == "w1":
        return w1old, manual, "w1old", "manual"
    return system_version(m, W3_OLD / f"{m}.json", "w3o")[0], w1old, "w3old", "w1old"


def load(version, col, m, n):
    d = json.loads((V2 / version / col / f"{m}.json").read_text())
    assert len(d["values"]) == n, f"{version}/{col}/{m}: {len(d['values'])} cached values, {n} segments"
    return d["values"], d["margins"]


def cells(workload, split="eval"):
    """replay.cells' record per changed cell, from the v2 caches; plus the meeting and the new segment's text."""
    out = []
    for m in meetings(split):
        old, new, on, nn = pair(workload, m)
        prov = provenance(old, new)
        kinds = {j: k for k, _, n in change_map(old.segs, new.segs) for j in n}
        for col in COLS:
            ov, om = load(on, col, m, len(old.segs))
            nv, _ = load(nn, col, m, len(new.segs))
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
                            "rule_conf": min(om[i] for i in contrib) if contrib else None,
                            "anchor_conf": om[a] if a is not None else None, "text": seg.text})
    return out


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(pa[k] * pb.get(k, 0) for k in pa) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def rule_agreement(cs):
    by = defaultdict(list)
    for c in cs:
        if c["rule"] not in (None, "hash"):
            by[(c["col"], c["rule"])].append((c["rule_val"], c["truth"]))
    return {f"{col}/{rule}": {"n": len(p), "rule_err": sum(a != b for a, b in p) / len(p),
                              "majority_err": 1 - Counter(t for _, t in p).most_common(1)[0][1] / len(p),
                              "kappa": kappa(p)} for (col, rule), p in sorted(by.items())}


def summarize(runs):
    calls = np.array([r[0] for r in runs], float)
    err = np.array([r[1] / r[2] for r in runs])
    viol = [any(s["violation"] for s in r[3].values()) for r in runs]
    return {"cells": runs[0][2], "call_share": float(calls.mean() / runs[0][2]), "calls_mean": float(calls.mean()),
            "cell_error_mean": float(err.mean()), "cell_error_max": float(err.max()),
            "any_stratum_violation_rate": float(np.mean(viol))}


def boot(cs, policy, eps, rng_boot):
    """Meeting bootstrap, whole policy re-run per resample with a fresh trial seed: 95% percentile CIs."""
    by_m = defaultdict(list)
    for c in cs:
        by_m[c["meeting"]].append(c)
    ms = sorted(by_m)
    share, err = [], []
    for b in range(BOOT):
        sample = [c for m in rng_boot.choice(ms, len(ms)) for c in by_m[m]]
        calls, wrong, n, _ = run_policy(policy, sample, np.random.default_rng([b, 11]), eps, DELTA)
        share.append(calls / n)
        err.append(wrong / n)
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"call_share_ci": q(share), "cell_error_ci": q(err)}


def flip(v, col, rng):
    c = COLUMNS[col]
    space = ["yes", "no"] if c.kind == "binary" else list(range(10)) if c.kind == "count" else list(c.options)
    return rng.choice([x for x in space if x != v])


def adversarial(cs, eps, rng_seed=0, min_n=50):
    """EchoBase strata with provisional values corrupted to a true error of eps + 0.02 (strata with N >= min_n);
    every certification is then a violation. Observed family-wise rate vs nominal delta, and per-stratum rate."""
    strata = defaultdict(list)
    for c in cs:
        if c["rule"] == "hash":
            continue
        if c["rule"] is not None:
            strata[(c["col"], "rule", c["rule"])].append(c)
        elif c["anchor_val"] is not None:
            strata[(c["col"], "anchor", c["change"])].append(c)
    alpha = DELTA / max(len(strata), 1)  # the same union bound as run_policy (all strata count)
    big = {k: v for k, v in strata.items() if len(v) >= min_n}
    fam, per = [], []
    for t in range(TRIALS):
        rng = np.random.default_rng([t, 23, rng_seed])
        any_v = False
        for key, items in big.items():
            truth = [c["truth"] for c in items]
            n_bad = math.ceil((eps + 0.02) * len(items))
            prov = list(truth)
            for i in rng.choice(len(items), n_bad, replace=False):
                prov[i] = flip(truth[i], key[0], rng)
            _, _, ok, _ = audit(prov, truth, rng, eps, alpha)
            per.append(ok)
            any_v |= ok
        fam.append(any_v)
    return {"strata_injected": len(big), "strata_total": len(strata), "alpha_per_stratum": alpha,
            "family_violation_rate": float(np.mean(fam)), "per_stratum_certify_rate": float(np.mean(per)),
            "nominal_delta": DELTA}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["w1", "w3"], default="w1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("eval maintenance replay")
    t0 = time.time()
    cs = cells(a.workload)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PRIMARY)
    tok_mean = float(np.mean([len(tok(c["text"])["input_ids"]) for c in cs if c["col"] == COLS[0]]))
    res = {"workload": a.workload, "split": "eval", "delta": DELTA, "trials": TRIALS, "boot": BOOT,
           "changed_cells": len(cs), "changed_cells_by_change": dict(Counter(c["change"] for c in cs if c["col"] == COLS[0])),
           "rule_share": dict(Counter(c["rule"] or "none" for c in cs)), "mean_tokens_changed_segment": tok_mean,
           "rules": rule_agreement(cs), "policies": {}, "adversarial": {}}
    for eps in EPS:
        for pol in POLICIES:
            if pol in ("full", "hash") and eps != EPS[0]:
                continue  # eps-independent
            runs = [run_policy(pol, cs, np.random.default_rng([t, 7]), eps, DELTA) for t in range(TRIALS)]
            r = summarize(runs)
            r["tokens_est"] = r["calls_mean"] * tok_mean
            r.update(boot(cs, pol, eps, np.random.default_rng([0, 5])))
            res["policies"][f"{pol}@{eps}" if pol not in ("full", "hash") else pol] = r
            print(f"{pol:13} eps={eps:.2f} share={r['call_share']:.3f} {r['call_share_ci']} "
                  f"err={r['cell_error_mean']:.4f} {r['cell_error_ci']} viol={r['any_stratum_violation_rate']:.3f}",
                  flush=True)
        res["adversarial"][str(eps)] = adversarial(cs, eps)
        print("adversarial", eps, res["adversarial"][str(eps)], flush=True)
    res["wall_clock_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1))
