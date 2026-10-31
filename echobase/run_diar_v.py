"""Diarization with a chosen pyannote pipeline version and clustering threshold (W3), known speaker count.

    python -m echobase.run_diar_v --pipeline 3.1 --threshold 0.55 --split dev --limit 4 --gpu 1
    python -m echobase.run_diar_v --pipeline 3.1 --estimate-speakers --split dev --limit 4 --gpu 1
Outputs outputs/diar_v/<pipeline>[-t<threshold>][-est]/<meeting>.json. W3 (exploratory, D3): old = 3.1 with an estimated
speaker count, new = 3.1 with the known count (D2's threshold variant changed 0.4% of segments). Do not run 3.0 on gpulab2: the host hard-reset twice as it started (2026-10-28).
Eval-split diarization is refused before G1 (plan v2).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from pilots.ami import audio_ready, meetings, speakers

from .gate import require_g1

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"3.0": "pyannote/speaker-diarization-3.0", "3.1": "pyannote/speaker-diarization-3.1"}

ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", required=True, choices=list(MODELS))
ap.add_argument("--split", default="dev", choices=["dev", "eval"])
ap.add_argument("--threshold", type=float, default=None)  # None = the pipeline's own clustering threshold
ap.add_argument("--estimate-speakers", action="store_true")  # let the pipeline choose the number of speakers
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gpu", type=int, default=0)
a = ap.parse_args()
if a.split == "eval":
    require_g1("eval diarization")

pipe = Pipeline.from_pretrained(MODELS[a.pipeline], use_auth_token=(Path.home() / ".hf_token").read_text().strip())
if a.threshold is not None:
    params = pipe.parameters(instantiated=True)
    params["clustering"]["threshold"] = a.threshold
    pipe.instantiate(params)
pipe.to(torch.device(f"cuda:{a.gpu}"))
tag = a.pipeline + (f"-t{a.threshold}" if a.threshold is not None else "") + ("-est" if a.estimate_speakers else "")
out_dir = ROOT / "outputs" / "diar_v" / tag
out_dir.mkdir(parents=True, exist_ok=True)
for m in meetings(a.split)[: a.limit or None]:
    out = out_dir / f"{m}.json"
    if out.exists() or not audio_ready(m):
        continue
    t0 = time.time()
    diar = pipe(str(ROOT / "data" / "ami" / "audio" / f"{m}.Mix-Headset.wav"), **({} if a.estimate_speakers else {"num_speakers": len(speakers(m))}))
    turns = [{"start": t.start, "end": t.end, "speaker": s} for t, _, s in diar.itertracks(yield_label=True)]
    out.write_text(json.dumps({"meeting": m, "model": MODELS[a.pipeline], "params": pipe.parameters(instantiated=True),
                               "turns": turns}))
    print(f"{tag} {m}: {len(turns)} turns, {time.time() - t0:.0f}s", flush=True)
