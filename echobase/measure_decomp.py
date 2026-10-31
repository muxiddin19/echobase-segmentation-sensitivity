"""Exploratory (AMI dev only): how many changed cells do EchoBase's rules derive, and how often do they agree
with a full recompute? Also caches every recomputed value (the reference) for later experiments.

    python -m echobase.measure_decomp --columns action_item,question --gpu 0 --out outputs/echobase/decomp_A.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pilots.ami import meetings
from pilots.segmap import change_map

from .lineage import provenance, versions
from .llm import COLUMNS, Scorer
from .rules import derive

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "echobase" / "values"

p = argparse.ArgumentParser()
p.add_argument("--columns", required=True)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--split", default="dev", choices=["dev"])   # eval meetings stay untouched until G1
p.add_argument("--out", required=True)
a = p.parse_args()

scorer = Scorer(gpu=a.gpu)


def values(column, tag, meeting, texts):
    f = CACHE / column / f"{meeting}.{tag}.json"
    if f.exists():
        return json.loads(f.read_text())
    v = scorer(column, texts)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(v))
    return v


report = {}
for column in a.columns.split(","):
    kind = COLUMNS[column].kind
    stats = defaultdict(Counter)   # change kind -> counts
    for m in meetings(a.split):
        old, new, _ = versions(m)
        prov = provenance(old, new)
        old_vals = values(column, "old", m, [s.text for s in old.segs])
        new_vals = values(column, "new", m, [s.text for s in new.segs])
        kinds = {}
        for k, _, n in change_map(old.segs, new.segs):
            for j in n:
                kinds[j] = k
        for j, seg in enumerate(new.segs):
            k = kinds.get(j, "new")
            if k == "unchanged":
                continue
            v, rule = derive(kind, seg.text, prov[j], old.segs, old_vals)
            c = stats[k]
            c["n"] += 1
            if v is not None:
                c["derived"] += 1
                c[f"rule:{rule}"] += 1
                c["agree"] += int(v == new_vals[j])
                c[f"agree:{rule}"] += int(v == new_vals[j])
        print(f"{column} {m} done (calls so far {scorer.calls})", flush=True)
    total = sum((c for c in stats.values()), Counter())
    report[column] = {
        "kind": kind, "by_change_kind": {k: dict(c) for k, c in stats.items()}, "total": dict(total),
        "derived_share": total["derived"] / total["n"] if total["n"] else 0.0,
        "agreement_on_derived": total["agree"] / total["derived"] if total["derived"] else None,
    }
    print(json.dumps({column: {k: report[column][k] for k in ("derived_share", "agreement_on_derived")}}), flush=True)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"model": scorer.id, "split": a.split, "llm_calls": scorer.calls, "columns": report}, indent=1))
