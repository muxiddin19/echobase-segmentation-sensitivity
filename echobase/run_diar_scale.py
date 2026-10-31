"""Post-hoc scale analysis (deviation D5): W1's old side for AMI meetings outside the dev and eval lists.

    HF_HUB_OFFLINE=1 python -m echobase.run_diar_scale --meetings configs/ami_scale.txt --gpu 0
The same pipeline and call as the frozen pilots/run_diar.py (pyannote 3.1, known speaker count); only the meeting list
differs, and the pipeline loads from the local model cache, so no access token is read. Output: outputs/diar/<m>.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from pilots.ami import audio_ready, speakers

ROOT = Path(__file__).resolve().parents[1]
MODEL = "pyannote/speaker-diarization-3.1"

p = argparse.ArgumentParser()
p.add_argument("--meetings", required=True)
p.add_argument("--gpu", type=int, default=0)
a = p.parse_args()

token_file = Path.home() / ".hf_token"
token = token_file.read_text().strip() if token_file.exists() and not os.environ.get("HF_HUB_OFFLINE") else None
pipe = Pipeline.from_pretrained(MODEL, use_auth_token=token)
pipe.to(torch.device(f"cuda:{a.gpu}"))

out_dir = ROOT / "outputs" / "diar"
out_dir.mkdir(parents=True, exist_ok=True)
ms = [l.strip() for l in Path(a.meetings).read_text().splitlines() if l.strip() and not l.startswith("#")]
for m in ms:
    out = out_dir / f"{m}.json"
    wav = ROOT / "data" / "ami" / "audio" / f"{m}.Mix-Headset.wav"
    if out.exists() or not audio_ready(m):
        continue
    t0 = time.time()
    diar = pipe(str(wav), num_speakers=len(speakers(m)))
    turns = [{"start": t.start, "end": t.end, "speaker": spk}
             for t, _, spk in diar.itertracks(yield_label=True)]
    out.write_text(json.dumps({"meeting": m, "model": MODEL, "turns": turns}))
    print(f"{m}: {len(turns)} turns, {time.time() - t0:.0f}s", flush=True)
