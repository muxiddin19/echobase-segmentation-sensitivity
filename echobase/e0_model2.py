"""E0.7 (plan v2): admit the second model. Binary yes-rate in [0.05, 0.60] on 500 dev segments, distinct answer tokens.

    python -m echobase.e0_model2 --model HuggingFaceTB/SmolLM2-1.7B-Instruct --gpu 0
Same 500 AMI dev (manual) segments as E0.1 (seed 0). Distinct first tokens are asserted by Scorer._targets for every
column. Agreement with the primary model's cached labels is reported (exploratory). Fallback: Qwen2.5-0.5B-Instruct.
"""
import argparse
import json
import random
from pathlib import Path

from pilots.ami import meetings

from .lineage import versions
from .llm import COLUMNS, Scorer

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "outputs" / "echobase" / "values"
BAND = (0.05, 0.60)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()

rng = random.Random(0)
pool = [(m, j, s.text) for m in meetings("dev") for j, s in enumerate(versions(m)[1].segs)]
sample = rng.sample(pool, a.n)
texts = [t for _, _, t in sample]
scorer = Scorer(model_id=a.model, gpu=a.gpu)
rep = {"model": a.model, "n": a.n, "columns": {}}
for col in ("action_item", "question", "entity_count", "sentiment", "topic"):
    scorer._targets(COLUMNS[col])  # raises if answer first-tokens collide
    vals = scorer(col, texts)
    ref = [json.loads((V / col / f"{m}.new.json").read_text())[j] for m, j, _ in sample]
    r = {"agree_with_primary": sum(x == y for x, y in zip(vals, ref)) / a.n,
         "distribution": {str(k): vals.count(k) / a.n for k in sorted(set(vals), key=str)}}
    if COLUMNS[col].kind == "binary":
        r["yes_rate"] = vals.count("yes") / a.n
        r["in_band"] = BAND[0] <= r["yes_rate"] <= BAND[1]
    rep["columns"][col] = r
rep["admitted"] = all(r.get("in_band", True) for r in rep["columns"].values())
out = ROOT / "reports" / f"e0_model2_{a.model.split('/')[-1]}.json"
out.write_text(json.dumps(rep, indent=1))
print(json.dumps({c: {k: v for k, v in r.items() if k != "distribution"} for c, r in rep["columns"].items()}),
      "admitted:", rep["admitted"])
