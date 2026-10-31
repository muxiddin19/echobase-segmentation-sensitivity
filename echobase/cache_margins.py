"""Cache (value, margin) for every old/new segment of the AMI dev meetings (exploratory input for e03+).

    python -m echobase.cache_margins --columns action_item,question,entity_count --gpu 0
Asserts that the labels equal the earlier label-only cache, so margins are added without changing any value.
"""
import argparse
import json
from pathlib import Path

from pilots.ami import meetings

from .lineage import versions
from .llm import Scorer

ROOT = Path(__file__).resolve().parents[1]
VALUES, MARGINS = ROOT / "outputs" / "echobase" / "values", ROOT / "outputs" / "echobase" / "margins"

p = argparse.ArgumentParser()
p.add_argument("--columns", required=True)
p.add_argument("--gpu", type=int, default=0)
a = p.parse_args()
scorer = Scorer(gpu=a.gpu)
for col in a.columns.split(","):
    for m in meetings("dev"):
        old, new, _ = versions(m)
        for tag, segs in (("old", old.segs), ("new", new.segs)):
            f = MARGINS / col / f"{m}.{tag}.json"
            if f.exists():
                continue
            vm = scorer.score(col, [s.text for s in segs])
            prev = json.loads((VALUES / col / f"{m}.{tag}.json").read_text())
            mism = sum(v != p for (v, _), p in zip(vm, prev))
            assert mism == 0, f"{col} {m} {tag}: {mism} labels differ from the cached run"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps([m_ for _, m_ in vm]))
        print(f"{col} {m} done", flush=True)
