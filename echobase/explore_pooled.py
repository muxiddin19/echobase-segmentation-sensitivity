"""Post-hoc exploratory (deviation D7): a pooled audit in the maintenance replay.

    PYTHONPATH=. python -m echobase.explore_pooled --workload w1 --out reports/explore_pooled_w1.json
anchor_cert (replay.run_policy) audits one stratum per column with a union bound over strata. The pooled policy puts
every non-identical cell with an anchor into a single stratum and audits it with the locked certify.audit at
alpha = delta. Its guarantee is weaker: the overall cell error is at most eps, while a single column may be worse.
Cells are e5_maintenance.cells (the locked replay records), so call shares are comparable with Tab. maint.
Reported per eps: call share of full recomputation, overall and per-column true error, and how often the pooled
stratum certified, over 100 trials.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .certify import audit
from .e5_maintenance import COLS, DELTA, EPS, TRIALS, cells
from .gate import require_g1


def pooled(cs, rng, eps, delta):
    """(llm_calls, wrong cells per column, certified)."""
    calls, items = 0, []
    for idx, c in enumerate(cs):
        if c["rule"] == "hash":
            continue  # identical text: reused exactly, like the hash baseline
        if c["anchor_val"] is None:
            calls += 1  # no anchor: must recompute
            continue
        items.append(idx)
    prov = [cs[i]["anchor_val"] for i in items]
    truth = [cs[i]["truth"] for i in items]
    final, n_calls, ok, _ = audit(prov, truth, rng, eps, delta)
    wrong = defaultdict(int)
    for i, f, t in zip(items, final, truth):
        wrong[cs[i]["col"]] += f != t
    return calls + n_calls, wrong, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=["w1", "w3"], default="w1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc pooled audit")
    cs = cells(a.workload)
    n_col = {c: sum(x["col"] == c for x in cs) for c in COLS}
    hash_share = sum(c["rule"] != "hash" for c in cs) / len(cs)
    anch = [c for c in cs if c["rule"] != "hash" and c["anchor_val"] is not None]
    anchor_error = sum(c["anchor_val"] != c["truth"] for c in anch) / len(anch)  # what the pooled audit must certify
    res = {"workload": a.workload, "changed_cells": len(cs), "delta": DELTA, "trials": TRIALS,
           "hash_share": hash_share, "anchored_cells": len(anch), "anchor_error": anchor_error, "eps": {}}
    print(f"{len(anch)} anchored cells, anchor error {anchor_error:.3f}", flush=True)
    for eps in EPS:
        runs = [pooled(cs, np.random.default_rng([t, 7]), eps, DELTA) for t in range(TRIALS)]
        shares = [r[0] / len(cs) for r in runs]
        err = [sum(r[1].values()) / len(cs) for r in runs]
        col_err = {c: float(np.mean([r[1][c] / n_col[c] for r in runs])) for c in COLS}
        res["eps"][str(eps)] = {"call_share": float(np.mean(shares)), "call_share_sd": float(np.std(shares)),
                                "cell_error_mean": float(np.mean(err)), "cell_error_max": float(np.max(err)),
                                "column_error_mean": col_err, "worst_column_error": max(col_err.values()),
                                "certified_rate": float(np.mean([r[2] for r in runs]))}
        r = res["eps"][str(eps)]
        print(f"eps={eps:.2f} share={r['call_share']:.3f} (hash {hash_share:.3f}) err={r['cell_error_mean']:.4f} "
              f"worst col={r['worst_column_error']:.4f} certified={r['certified_rate']:.2f}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
