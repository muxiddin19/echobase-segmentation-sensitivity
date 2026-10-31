"""Pilot A (locked evaluation loop, program_pilotA.md): strong baseline vs full recompute under re-segmentation.

    python -m pilots.run_pilotA --condition structural --gpu 0 --out runs/pilotA/structural/s0
    python -m pilots.run_pilotA --condition structural --gpu 0 --limit 2 --out runs/smoke/...   # smoke tier

Writes <out>/result.json (counts per meeting, pooled rates, bootstrap CIs, verdict for the structural condition).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .ami import meetings
from .build_versions import version0, version1
from .extract import Extractor
from .metrics import STRUCTURAL, baseline_decisions, evaluate, self_disagreement
from .segmap import change_map
from .stats import bootstrap, pooled_rates
from .verdict import verdict_pilotA

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COLUMN = "action_item"

p = argparse.ArgumentParser()
p.add_argument("--condition", required=True, choices=["structural", "asr"])
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="smoke tier only: first N dev meetings")
p.add_argument("--out", required=True)
a = p.parse_args()

ms = meetings("dev")[: a.limit or None]
missing = [m for m in ms if not (ROOT / "outputs" / "diar" / f"{m}.json").exists()
           or not (ROOT / "outputs" / "asr" / "large-v3" / f"{m}.json").exists()]
if missing:
    raise SystemExit(f"upstream outputs missing for {missing}")

ex = Extractor(MODEL, gpu=a.gpu, dtype=torch.float32)
kinds, per_meeting, floor = Counter(), {}, ([], [])
for m in ms:
    old, new = version0(m, a.condition), version1(m)
    cmap = change_map(old, new)
    kinds.update(k for k, _, n in cmap for _ in (n or [None]))
    old_vals = [lab for lab, _ in ex(COLUMN, [s.text for s in old])]
    new_vals = [lab for lab, _ in ex(COLUMN, [s.text for s in new])]
    dec = baseline_decisions(old, new)
    changed = [j for k, _, n in cmap if k != "unchanged" for j in n]
    structural = [j for k, _, n in cmap if k in STRUCTURAL for j in n]
    per_meeting[m] = {"n_old": len(old), "n_new": len(new),
                      "changed": evaluate(old, new, old_vals, new_vals, dec, changed),
                      "structural": evaluate(old, new, old_vals, new_vals, dec, structural)}
    # noise floor: the same old texts labelled again in reverse order (different batch composition)
    again = [lab for lab, _ in ex(COLUMN, [s.text for s in old][::-1])][::-1]
    floor[0].extend(old_vals)
    floor[1].extend(again)
    print(f"{m}: old={len(old)} new={len(new)} structural={len(structural)}", flush=True)

n_changed = sum(v for k, v in kinds.items() if k != "unchanged")
share = sum(kinds[k] for k in STRUCTURAL) / n_changed if n_changed else 0.0
rows = [v["structural"] for v in per_meeting.values()]
point = pooled_rates(rows)
ci = bootstrap(rows)
fl = self_disagreement(*floor)
result = {
    "condition": a.condition, "model": ex.id, "column": COLUMN, "n_meetings": len(per_meeting),
    "change_kinds_by_new_segment": dict(kinds), "share": share,
    "structural_counts": {k: sum(r[k] for r in rows) for k in ("n", "reused", "wrong", "recomputed", "value_changed")},
    "W": point["W"], "R": point["R"], "W_ci95": ci["W"], "R_ci95": ci["R"], "floor": fl,
    "changed_counts": {k: sum(v["changed"][k] for v in per_meeting.values())
                       for k in ("n", "reused", "wrong", "recomputed", "value_changed")},
    "per_meeting": per_meeting,
}
result["verdict"] = verdict_pilotA(result) if a.condition == "structural" and not a.limit else "n/a"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k != "per_meeting"}, indent=1))
