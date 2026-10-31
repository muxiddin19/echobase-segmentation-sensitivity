"""Reviewer issue 1 & 6 check (exploratory, AMI dev): paired anchor-vs-rule error on identical cells, plus
chance-corrected agreement (Cohen's kappa) and majority-class baseline per rule."""
from collections import Counter

from .replay import cells, paired

COLS = "action_item,question,entity_count,sentiment,topic".split(",")
cs = cells(COLS)


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(pa[k] * pb.get(k, 0) for k in pa) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


print(f"{'column/rule':28} {'N':>5} {'rule_err':>8} {'anchor_err':>10} {'same_value':>10} {'majority_err':>12} {'kappa_rule':>10}")
for (col, rule), r in sorted(paired(cs).items(), key=lambda kv: -kv[1]["n"]):
    p, n = r["pairs"], r["n"]
    same = sum(rv == av for rv, av, _ in p) / n
    maj = Counter(t for _, _, t in p).most_common(1)[0][1] / n
    k = kappa([(rv, t) for rv, _, t in p])
    print(f"{col + '/' + rule:28} {n:5d} {r['rule_err']:8.3f} {r['anchor_err']:10.3f} {same:10.3f} {1 - maj:12.3f} {k:10.3f}")

print("--- cells where no rule fires (not hash): anchor error and what share of changed cells they are")
rest = [c for c in cs if c["rule"] is None and c["anchor_val"] is not None]
tot = Counter(c["col"] for c in cs)
for col in COLS:
    xs = [c for c in rest if c["col"] == col]
    if xs:
        print(f"  {col:13} N={len(xs):5} share_of_changed={len(xs) / tot[col]:.3f} "
              f"anchor_err={sum(x['anchor_val'] != x['truth'] for x in xs) / len(xs):.3f}")
