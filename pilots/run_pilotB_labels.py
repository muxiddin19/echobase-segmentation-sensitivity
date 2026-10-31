"""Pilot B (locked, program_pilotB.md): both predicates scored on ONE shared Whisper transcript per dialogue act.

    python -m pilots.run_pilotB_labels --gpu 1 --out runs/pilotB/labels/s0 [--limit 2]

Writes <out>/units.json: per dialogue act the gold labels (human), the two LLM margins, ASR confidence, empty flag.
"""
import argparse
import json
from pathlib import Path

import torch

from .ami import dialogue_acts, meetings
from .build_versions import normalize
from .extract import Extractor

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GOLD1 = {"sug"}
GOLD2 = {"el.inf", "el.ass"}

p = argparse.ArgumentParser()
p.add_argument("--gpu", type=int, default=1)
p.add_argument("--limit", type=int, default=0, help="smoke tier only: first N dev meetings")
p.add_argument("--out", required=True)
a = p.parse_args()

units = []
for m in meetings("dev")[: a.limit or None]:
    words = json.loads((ROOT / "outputs" / "asr" / "large-v3" / f"{m}.json").read_text())["words"]
    for da in dialogue_acts(m):
        if da["type"] is None:
            continue
        ws = [w for w in words if da["start"] <= (w["start"] + w["end"]) / 2 <= da["end"]]
        units.append({
            "meeting": m, "id": da["id"], "type": da["type"],
            "gold1": da["type"] in GOLD1, "gold2": da["type"] in GOLD2,
            "text": normalize(" ".join(w["word"] for w in ws)),
            "asr_conf": sum(w["prob"] for w in ws) / len(ws) if ws else 0.0,
            "empty": not ws,
        })

ex = Extractor(MODEL, gpu=a.gpu, dtype=torch.float32)
texts = [u["text"] for u in units]
for key, column in (("margin1", "suggestion"), ("margin2", "elicit")):
    for u, (_, margin) in zip(units, ex(column, texts)):
        u[key] = margin
    print(f"{column}: done", flush=True)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "units.json").write_text(json.dumps({"model": ex.id, "n_meetings": len({u['meeting'] for u in units}),
                                            "units": units}))
pos = sum(u["gold1"] or u["gold2"] for u in units)
print(f"{len(units)} units, {pos} gold positives, {sum(u['empty'] for u in units)} empty transcripts")
