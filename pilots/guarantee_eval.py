"""Locked measurement (program_pilotB.md): recall-guaranteed retrieval under a label budget, two samplers.

    python -m pilots.guarantee_eval --units runs/pilotB/labels/s0/units.json --method lineage --seed 0 --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

TARGET, DELTA, K, TRIALS = 0.9, 0.1, 50, 100
DELTA_BOUND = 0.05  # the bound is built at 0.05 and judged against DELTA = 0.1 (normal approximation)
BUDGETS = [800, 1120, 1440, 1600]


def prepare(units):
    s = np.array([max(u["margin1"], u["margin2"]) for u in units])
    g = np.array([u["gold1"] or u["gold2"] for u in units])
    grid = np.unique(np.quantile(s, np.linspace(0, 1, K)))  # low tau (large set) first: fixed-sequence order
    return s, g, grid


def strata(units, s):
    """Lineage strata from system outputs only: ASR-confidence tercile x returned-at-0 x empty transcript."""
    conf = np.array([u["asr_conf"] for u in units])
    empty = np.array([u["empty"] for u in units])
    cuts = np.quantile(conf[~empty], [1 / 3, 2 / 3]) if (~empty).any() else [0, 0]
    tert = np.digitize(conf, cuts)
    h = np.where(empty, 6, tert * 2 + (s >= 0))
    risk = {k: 1 - conf[h == k].mean() if k != 6 else 1.0 for k in np.unique(h)}
    return h, risk


MIN_POS = 30  # no certificate from fewer sampled gold positives (normal approximation)
Z = float(norm.ppf(1 - DELTA_BOUND))


def recall_lcb(s, g, sizes, idx_by_h, tau):
    """Stratified ratio estimator of recall(tau) with a linearised variance and a one-sided normal bound.

    Both methods use this estimator; 'global' is the one-stratum case with a uniform sample, so the
    comparison isolates the sampling design. Finite-population correction per stratum.
    """
    tp = pos = 0.0
    for k, idx in idx_by_h.items():
        tp += sizes[k] * (g[idx] & (s[idx] >= tau)).mean()
        pos += sizes[k] * g[idx].mean()
    if pos <= 0:
        return 0.0
    r = tp / pos
    var = 0.0
    for k, idx in idx_by_h.items():
        n, N = len(idx), sizes[k]
        if n < 2:
            continue
        z = (g[idx] & (s[idx] >= tau)).astype(float) - r * g[idx].astype(float)
        var += N * N * (1 - n / N) * z.var(ddof=1) / n
    return r - Z * np.sqrt(var) / pos


# Fixed-sequence testing: recall(tau) is non-increasing in tau, so test thresholds from the largest set to the
# smallest, each at the full delta, and stop at the first one that does not certify (no /K correction).
def certify(s, g, sizes, idx_by_h, grid):
    if sum(int(g[idx].sum()) for idx in idx_by_h.values()) < MIN_POS:
        return None
    best = None
    for tau in grid:
        if recall_lcb(s, g, sizes, idx_by_h, tau) < TARGET:
            break
        best = tau
    return best


def allocate(sizes, risk, n):
    w = {k: sizes[k] * np.sqrt(risk[k]) for k in sizes}
    tot = sum(w.values())
    alloc = {k: min(sizes[k], max(2, int(round(n * w[k] / tot)))) for k in sizes}
    while sum(alloc.values()) > n:  # trim rounding overshoot from the largest stratum
        k = max(alloc, key=alloc.get)
        alloc[k] -= 1
    return alloc


def outcome(s, g, tau):
    R = np.ones_like(g) if tau is None else (s >= tau)
    tp = int((R & g).sum())
    return tp / max(int(R.sum()), 1), tp / max(int(g.sum()), 1)


def run(units, method, seed):
    s, g, grid = prepare(units)
    h, risk = strata(units, s)
    sizes = {k: int((h == k).sum()) for k in np.unique(h)}
    members = {k: np.flatnonzero(h == k) for k in sizes}
    res = {}
    for n in BUDGETS:
        rng = np.random.default_rng([seed, n, 0 if method == "global" else 1])
        prec, rec = [], []
        for _ in range(TRIALS):
            if method == "global":
                tau = certify(s, g, {0: len(s)}, {0: rng.choice(len(s), size=n, replace=False)}, grid)
            else:
                alloc = allocate(sizes, risk, n)
                idx_by_h = {k: rng.choice(members[k], size=alloc[k], replace=False) for k in sizes}
                tau = certify(s, g, sizes, idx_by_h, grid)
            p, r = outcome(s, g, tau)
            prec.append(p)
            rec.append(r)
        res[n] = {"prec": prec, "recall": rec, "viol": [x < TARGET for x in rec]}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--method", required=True, choices=["global", "lineage"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.units).read_text())
    res = run(data["units"], a.method, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {n: {"prec": float(np.mean(v["prec"])), "viol": float(np.mean(v["viol"]))} for n, v in res.items()}
    (out / "result.json").write_text(json.dumps({"method": a.method, "seed": a.seed, "n_meetings": data["n_meetings"],
                                                 "summary": summary, "trials": res}))
    print(json.dumps(summary, indent=1))
