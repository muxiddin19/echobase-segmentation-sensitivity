"""e03 (exploratory, AMI dev): does stratifying by upstream confidence let more strata certify?"""
from collections import defaultdict

import numpy as np

from .replay import cells, conf_bin, run_policy

COLS = "action_item,question,entity_count,sentiment,topic".split(",")
cs = cells(COLS)
n = len(cs)
for eps in (0.02, 0.05, 0.10):
    for pol in ("hash", "echobase", "echobase_conf"):
        r = [run_policy(pol, cs, np.random.default_rng([t, 7]), eps, 0.05) for t in range(10)]
        share = np.mean([x[0] for x in r]) / n
        err = np.mean([x[1] / x[2] for x in r])
        viol = np.mean([any(s["violation"] for s in x[3].values()) for x in r])
        print(f"eps={eps:.2f} {pol:14} call_share={share:.3f} err_mean={err:.4f} viol={viol:.2f}")

print("--- echobase_conf strata, largest first (eps=0.05, trial 0)")
_, _, _, info = run_policy("echobase_conf", cs, np.random.default_rng([0, 7]), 0.05, 0.05)
st = defaultdict(lambda: [0, 0])
for c in cs:
    if c["rule"] == "hash":
        continue
    if c["rule"]:
        pv, key = c["rule_val"], (c["col"], "rule", c["rule"], conf_bin(c["rule_conf"]))
    else:
        pv, key = c["anchor_val"], (c["col"], "anchor", c["change"], conf_bin(c["anchor_conf"]))
    if pv is None:
        continue
    k = "/".join(key)
    st[k][0] += 1
    st[k][1] += int(pv != c["truth"])
for k, (cnt, w) in sorted(st.items(), key=lambda x: -x[1][0])[:22]:
    print(f"  {k:40} N={cnt:5} prov_err={w / cnt:.3f} certified={info[k]['certified']}")
