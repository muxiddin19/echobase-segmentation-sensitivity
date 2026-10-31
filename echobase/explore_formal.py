"""Post-hoc (deviation D6), CPU only, AMI eval, primary model:

    PYTHONPATH=. python -m echobase.explore_formal --out reports/explore_formal.json

1. Merge monotonicity. A binary column f is monotone under concatenation if f(s+t) >= max(f(s), f(t)) (yes > no). For
   such f, merging segments can never lower the yes-share (Proposition 2 in the paper). Measured on manual -> merge{k}:
   per merged segment with >= 2 parts, 'loss' = some part yes but the merged segment no (violates monotonicity),
   'emergence' = all parts no but merged yes (allowed by monotonicity, not by OR-decomposability).
2. Audit power curve. For the W1 provenance strata with N >= 50, provisional values are corrupted to a true error
   e = eps + d, d in {-0.05, ..., +0.05}; the probability that the audit certifies a stratum is measured over 100
   trials (alpha as in the replay: delta / #strata). Certification with e > eps is a violation; with e <= eps a success.
3. Anchor accuracy per column on W1's changed cells (how often the most-overlapping old segment's value is right).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilots.ami import meetings

from .certify import audit
from .e5_maintenance import cells, flip
from .gate import require_g1
from .llm import PRIMARY, cache_root
from .score_versions import build

ROOT = Path(__file__).resolve().parents[1]
V = cache_root(ROOT, PRIMARY)
BINARY = ("action_item", "question")
DELTA, TRIALS = 0.05, 100


def monotonicity():
    out = {}
    for k in (2, 4, 8):
        for col in BINARY:
            loss = emerge = multi = 0
            for m in meetings("eval"):
                man, mer = build("manual", m), build(f"merge{k}", m)
                mv = json.loads((V / "manual" / col / f"{m}.json").read_text())["values"]
                gv = json.loads((V / f"merge{k}" / col / f"{m}.json").read_text())["values"]
                parts = defaultdict(set)
                for wi, j in mer.word_seg.items():
                    if wi in man.word_seg:
                        parts[j].add(man.word_seg[wi])
                for j, ps in parts.items():
                    if len(ps) < 2:
                        continue
                    multi += 1
                    any_yes = any(mv[i] == "yes" for i in ps)
                    if any_yes and gv[j] == "no":
                        loss += 1
                    elif not any_yes and gv[j] == "yes":
                        emerge += 1
            out[f"k{k}/{col}"] = {"merged_segments": multi, "loss": loss, "emergence": emerge,
                                  "loss_rate": loss / multi, "emergence_rate": emerge / multi}
    return out


def power_curve(cs, eps_list=(0.05, 0.10, 0.15)):
    strata = defaultdict(list)
    for c in cs:
        if c["rule"] == "hash":
            continue
        if c["rule"] is not None:
            strata[(c["col"], "rule", c["rule"])].append(c)
        elif c["anchor_val"] is not None:
            strata[(c["col"], "anchor", c["change"])].append(c)
    alpha = DELTA / max(len(strata), 1)
    big = {k: v for k, v in strata.items() if len(v) >= 50}
    out = {}
    for eps in eps_list:
        for d in (-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
            e = eps + d
            if e < 0:
                continue
            cert, viol, frac = [], [], []
            for t in range(TRIALS):
                rng = np.random.default_rng([t, 29])
                for key, items in big.items():
                    truth = [c["truth"] for c in items]
                    prov = list(truth)
                    n_bad = round(e * len(items))
                    for i in rng.choice(len(items), n_bad, replace=False):
                        prov[i] = flip(truth[i], key[0], rng)
                    _, calls, ok, _ = audit(prov, truth, rng, eps, alpha)
                    cert.append(ok)
                    viol.append(ok and n_bad / len(items) > eps)  # certified although the actual error exceeds eps
                    frac.append(calls / len(items))               # share of the stratum the audit recomputed
            out[f"{eps:.2f}/{d:+.2f}"] = {"eps": eps, "true_error": e, "certify_rate": float(np.mean(cert)),
                                          "violation_rate": float(np.mean(viol)), "sampled_share": float(np.mean(frac)),
                                          "strata": len(big), "alpha_per_stratum": alpha}
    return out


def anchor_accuracy(cs):
    by = defaultdict(lambda: [0, 0])
    for c in cs:
        if c["anchor_val"] is not None and c["rule"] != "hash":
            by[c["col"]][0] += c["anchor_val"] == c["truth"]
            by[c["col"]][1] += 1
    return {col: {"n": n, "accuracy": ok / n} for col, (ok, n) in by.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-power", action="store_true", help="re-run the power curve, keep the rest of --out")
    a = ap.parse_args()
    require_g1("post-hoc formal analyses")
    if a.only_power:
        res = json.loads(Path(a.out).read_text())
        res["power_curve"] = power_curve(cells("w1"))
        for k, r in res["power_curve"].items():
            print(k, round(r["certify_rate"], 3), round(r["violation_rate"], 4), round(r["sampled_share"], 3), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        raise SystemExit
    res = {"monotonicity": monotonicity()}
    print(json.dumps(res["monotonicity"], indent=1), flush=True)
    cs = cells("w1")
    res["anchor_accuracy"] = anchor_accuracy(cs)
    print(res["anchor_accuracy"], flush=True)
    res["power_curve"] = power_curve(cs)
    for k, r in res["power_curve"].items():
        print(k, round(r["certify_rate"], 3), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
