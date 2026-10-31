"""Whisper ASR with word timestamps for AMI Mix-Headset audio (the "system" transcript, version 0).

    python -m pilots.run_asr --split dev --gpu 0
"""
import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

from .ami import audio_ready, meetings

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--split", default="dev", choices=["dev", "eval"])
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--model", default="large-v3")
p.add_argument("--compute", default="int8")
a = p.parse_args()

out_dir = ROOT / "outputs" / "asr" / a.model
out_dir.mkdir(parents=True, exist_ok=True)
model = WhisperModel(a.model, device="cuda", device_index=a.gpu, compute_type=a.compute)

for m in meetings(a.split):
    out = out_dir / f"{m}.json"
    wav = ROOT / "data" / "ami" / "audio" / f"{m}.Mix-Headset.wav"
    if out.exists() or not audio_ready(m):
        continue
    t0 = time.time()
    # temperature=0 disables the stochastic fallback, so reruns give the same transcript
    segments, info = model.transcribe(str(wav), language="en", beam_size=5, temperature=0.0,
                                      word_timestamps=True, vad_filter=True,
                                      # without this, long meetings drift into skipping speech (ES2011a lost ~75% of words after minute 10)
                                      condition_on_previous_text=False)
    words = [{"start": w.start, "end": w.end, "word": w.word.strip(), "prob": w.probability}
             for s in segments for w in (s.words or [])]
    out.write_text(json.dumps({"meeting": m, "model": a.model, "compute": a.compute,
                               "duration": info.duration, "words": words}))
    print(f"{m}: {len(words)} words, {info.duration / 60:.1f} min audio, {time.time() - t0:.0f}s", flush=True)
