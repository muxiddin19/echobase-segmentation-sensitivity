"""Post-hoc check (deviation D7): do two hosts give the same labels for the same model and inputs?

    PYTHONPATH=. python -m echobase.explore_crosshost --a v2 --b v2-gp231-check --out reports/post_crosshost_1.5b.json
Compares every cached label of the AMI eval manual version in two cache dirs under outputs/echobase (here: the primary
model on gpulab2, used by the confirmatory analysis, and the same model rescored on gp231 before its scale run).
"""
import argparse
import json
from pathlib import Path

from pilots.ami import meetings

ROOT = Path(__file__).resolve().parents[1]
EB = ROOT / "outputs" / "echobase"
COLS = ("action_item", "question", "entity_count", "sentiment", "topic")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--version", default="manual")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"a": a.a, "b": a.b, "version": a.version, "columns": {}}
    for col in COLS:
        n = mism = 0
        for m in meetings("eval"):
            va = json.loads((EB / a.a / a.version / col / f"{m}.json").read_text())["values"]
            vb = json.loads((EB / a.b / a.version / col / f"{m}.json").read_text())["values"]
            assert len(va) == len(vb), f"{col}/{m}: {len(va)} vs {len(vb)} segments"
            n += len(va)
            mism += sum(x != y for x, y in zip(va, vb))
        res["columns"][col] = {"n": n, "mismatches": mism}
    res["cells"] = sum(r["n"] for r in res["columns"].values())
    res["mismatches"] = sum(r["mismatches"] for r in res["columns"].values())
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(res)
