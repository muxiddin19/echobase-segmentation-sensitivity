"""Build version 0 (system: diarization turns + Whisper words) and version 1 (AMI manual) segment lists.

Both sides get the same text normalisation, and system speaker labels are mapped to AMI speaker
letters by maximum total overlap. Otherwise every segment would count as a "relabel".
"""
import json
import re
from itertools import permutations
from pathlib import Path

from .ami import gold_segments, reference_words
from .segmap import Segment, overlap

ROOT = Path(__file__).resolve().parents[1]
MERGE_GAP = 0.5  # seconds; consecutive same-speaker turns closer than this form one system segment


def normalize(text):
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def _merge_turns(turns):
    merged = []
    for t in sorted(turns, key=lambda t: t["start"]):
        last = next((m for m in reversed(merged) if m["speaker"] == t["speaker"]), None)
        if last and t["start"] - last["end"] < MERGE_GAP and last is merged[-1]:
            last["end"] = max(last["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def _map_speakers(turns, gold):
    sys_spk = sorted({t["speaker"] for t in turns})
    gold_spk = sorted({g.speaker for g in gold})
    score = {(s, g): sum(overlap(Segment("", "", t["start"], t["end"], ""), x)
                         for t in turns if t["speaker"] == s for x in gold if x.speaker == g)
             for s in sys_spk for g in gold_spk}
    best, best_score = {}, -1.0
    k = min(len(sys_spk), len(gold_spk))
    for perm in permutations(gold_spk, k):
        sc = sum(score[(s, g)] for s, g in zip(sys_spk[:k], perm))
        if sc > best_score:
            best, best_score = dict(zip(sys_spk[:k], perm)), sc
    return {s: best.get(s, f"X{s}") for s in sys_spk}


def version1(meeting):
    return [Segment(s.sid, s.speaker, s.start, s.end, normalize(s.text)) for s in gold_segments(meeting)]


def version0(meeting, condition="asr", asr_model="large-v3"):
    """System segmentation (pyannote turns, speakers mapped to AMI letters) filled with words.

    condition="asr":        Whisper words  -> segmentation AND transcript differ from version 1
    condition="structural": AMI reference words -> only segmentation/speaker attribution differ
    """
    turns = json.loads((ROOT / "outputs" / "diar" / f"{meeting}.json").read_text())["turns"]
    spk_map = _map_speakers(turns, gold_segments(meeting))
    segs = [dict(s, speaker=spk_map[s["speaker"]]) for s in _merge_turns(turns)]
    if condition == "asr":
        words = [dict(w, speaker=None) for w in
                 json.loads((ROOT / "outputs" / "asr" / asr_model / f"{meeting}.json").read_text())["words"]]
    else:
        words = reference_words(meeting)
    texts = [[] for _ in segs]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        hits = [i for i, s in enumerate(segs) if s["start"] <= mid <= s["end"]]
        if w["speaker"]:  # reference words: prefer a covering segment of the word's own speaker
            hits = [i for i in hits if segs[i]["speaker"] == w["speaker"]] or hits
        if hits:  # overlapping turns: give the word to the shortest covering turn
            i = min(hits, key=lambda i: segs[i]["end"] - segs[i]["start"])
            texts[i].append(w["word"])
    return [Segment(f"{meeting}.sys.{i}", s["speaker"], s["start"], s["end"], normalize(" ".join(t)))
            for i, (s, t) in enumerate(zip(segs, texts)) if t]
