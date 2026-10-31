"""pyannote speaker diarization for AMI Mix-Headset audio (system speaker turns, version 0).

    python -m pilots.run_diar --split dev --gpu 1
"""
import argparse
import json
import time
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from .ami import audio_ready, meetings, speakers

ROOT = Path(__file__).resolve().parents[1]
MODEL = "pyannote/speaker-diarization-3.1"

p = argparse.ArgumentParser()
p.add_argument("--split", default="dev", choices=["dev", "eval"])
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--reverse", action="store_true", help="walk the meeting list backwards (second worker)")
a = p.parse_args()

token = (Path.home() / ".hf_token").read_text().strip()
pipe = Pipeline.from_pretrained(MODEL, use_auth_token=token)
pipe.to(torch.device(f"cuda:{a.gpu}"))

out_dir = ROOT / "outputs" / "diar"
out_dir.mkdir(parents=True, exist_ok=True)
for m in (reversed if a.reverse else list)(meetings(a.split)):
    out = out_dir / f"{m}.json"
    wav = ROOT / "data" / "ami" / "audio" / f"{m}.Mix-Headset.wav"
    if out.exists() or not audio_ready(m):
        continue
    try:  # claim the meeting so two workers never process the same one
        (out_dir / f"{m}.lock").open("x").close()
    except FileExistsError:
        continue
    t0 = time.time()
    # the number of speakers is known from the corpus metadata; the system still has to find who speaks when
    diar = pipe(str(wav), num_speakers=len(speakers(m)))
    turns = [{"start": t.start, "end": t.end, "speaker": spk}
             for t, _, spk in diar.itertracks(yield_label=True)]
    out.write_text(json.dumps({"meeting": m, "model": MODEL, "turns": turns}))
    print(f"{m}: {len(turns)} turns, {time.time() - t0:.0f}s", flush=True)
