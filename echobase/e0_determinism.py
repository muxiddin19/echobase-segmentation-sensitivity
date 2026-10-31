"""E0.1 (plan v2): is the scorer deterministic under shuffled order, different batch budgets and another GPU?

    python -m echobase.e0_determinism --gpu 0 --n 500 --out reports/e0_determinism_gpu0.json
Reference = the cached labels/margins from the original run: dev = outputs/echobase/values, margins; eval (the
PREREGISTRATION §6 spot check, --split eval --n 100) = the E2 cache outputs/echobase/v2/manual.
"""
import argparse
import json
import random
from pathlib import Path

from pilots.ami import meetings

from .gate import require_g1
from .lineage import versions
from .llm import Scorer

ROOT = Path(__file__).resolve().parents[1]
V, M = ROOT / "outputs" / "echobase" / "values", ROOT / "outputs" / "echobase" / "margins"
V2 = ROOT / "outputs" / "echobase" / "v2" / "manual"

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--split", default="dev", choices=["dev", "eval"])
ap.add_argument("--columns", default="action_item,entity_count,topic")
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.split == "eval":
    require_g1("the eval determinism spot check")

rng = random.Random(0)
pool = []
for m in meetings(a.split):
    _, new, _ = versions(m)
    for j, s in enumerate(new.segs):
        pool.append((m, j, s.text))
sample = rng.sample(pool, a.n)
scorer = Scorer(gpu=a.gpu)
report = {"gpu": a.gpu, "n": a.n, "columns": {}}
for col in a.columns.split(","):
    if a.split == "eval":
        cache = {m: json.loads((V2 / col / f"{m}.json").read_text()) for m in {m for m, _, _ in sample}}
        ref_v = [cache[m]["values"][j] for m, j, _ in sample]
        ref_m = [cache[m]["margins"][j] for m, j, _ in sample]
    else:
        ref_v = [json.loads((V / col / f"{m}.new.json").read_text())[j] for m, j, _ in sample]
        ref_m = [json.loads((M / col / f"{m}.new.json").read_text())[j] for m, j, _ in sample]
    runs = {}
    for budget in (1024, 3072, 6144):
        order = list(range(a.n))
        rng.shuffle(order)
        res = scorer.score(col, [sample[i][2] for i in order], token_budget=budget)
        vals, margs = [None] * a.n, [None] * a.n
        for i, (v, mg) in zip(order, res):
            vals[i], margs[i] = v, mg
        runs[budget] = {"label_mismatches": sum(x != y for x, y in zip(vals, ref_v)),
                        "max_margin_drift": max(abs(x - y) for x, y in zip(margs, ref_m))}
    report["columns"][col] = runs
    print(col, runs, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(report, indent=1))
