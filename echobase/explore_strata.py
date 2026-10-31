"""Exploratory (AMI dev): epsilon sweep for the echobase policy and the true provisional error per stratum."""
from collections import defaultdict

import numpy as np

from .replay import cells, run_policy

COLS = "action_item,question,entity_count,sentiment,topic".split(",")
cs = cells(COLS)
for eps in (0.02, 0.05, 0.10, 0.15):
    r = [run_policy("echobase", cs, np.random.default_rng([t, 7]), eps, 0.05) for t in range(10)]
    share = np.mean([x[0] for x in r]) / r[0][2]
    err = np.mean([x[1] / x[2] for x in r])
    viol = np.mean([any(s["violation"] for s in x[3].values()) for x in r])
    print(f"eps={eps:.2f} call_share={share:.3f} err_mean={err:.4f} viol={viol:.2f}")

print("--- strata (provisional error = share of provisional values a recompute would change)")
_, _, _, info = run_policy("echobase", cs, np.random.default_rng([0, 7]), 0.05, 0.05)
st = defaultdict(lambda: [0, 0])
for c in cs:
    if c["rule"] == "hash":
        continue
    if c["rule"]:
        pv, key = c["rule_val"], (c["col"], "rule", c["rule"])
    else:
        pv, key = c["anchor_val"], (c["col"], "anchor", c["change"])
    if pv is None:
        continue
    k = "/".join(key)
    st[k][0] += 1
    st[k][1] += int(pv != c["truth"])
for k, (n, w) in sorted(st.items(), key=lambda x: -x[1][0]):
    print(f"  {k:34} N={n:5} prov_err={w / n:.3f} certified@0.05={info[k]['certified']}")
