"""Word-level lineage between two segmentations of the same words (structural condition).

For each new segment: which old segments contributed its words, which of them it contains completely, and
whether every one of its words came from some old segment. This is the provenance EchoBase's rules read.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from pilots.ami import gold_segments, reference_words
from pilots.build_versions import _map_speakers, _merge_turns, normalize
from pilots.segmap import Segment

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Version:
    segs: list                                     # Segment objects with normalized text
    word_seg: dict = field(default_factory=dict)   # word index -> segment index


def _assign(words, segs, prefer_speaker=True):
    out = {}
    for wi, w in enumerate(words):
        mid = (w["start"] + w["end"]) / 2
        hits = [i for i, s in enumerate(segs) if s["start"] <= mid <= s["end"]]
        if prefer_speaker:
            hits = [i for i in hits if segs[i]["speaker"] == w["speaker"]] or hits
        if hits:
            out[wi] = min(hits, key=lambda i: segs[i]["end"] - segs[i]["start"])
    return out


def _build(raw, words, assign, sid_prefix):
    texts = [[] for _ in raw]
    for wi in sorted(assign, key=lambda wi: words[wi]["start"]):
        texts[assign[wi]].append(words[wi]["word"])
    keep = [i for i, t in enumerate(texts) if t]
    remap = {old: new for new, old in enumerate(keep)}
    segs = [Segment(f"{sid_prefix}{i}", raw[i]["speaker"], raw[i]["start"], raw[i]["end"],
                    normalize(" ".join(texts[i]))) for i in keep]
    return Version(segs, {wi: remap[s] for wi, s in assign.items()})


def versions(meeting):
    """(old, new, words): system segmentation vs AMI manual segmentation over the same reference words."""
    words = reference_words(meeting)
    turns = json.loads((ROOT / "outputs" / "diar" / f"{meeting}.json").read_text())["turns"]
    gold = gold_segments(meeting)
    spk = _map_speakers(turns, gold)
    old_raw = [dict(t, speaker=spk[t["speaker"]]) for t in _merge_turns(turns)]
    new_raw = [{"start": g.start, "end": g.end, "speaker": g.speaker} for g in gold]
    old = _build(old_raw, words, _assign(words, old_raw), f"{meeting}.sys.")
    new = _build(new_raw, words, _assign(words, new_raw), f"{meeting}.gold.")
    return old, new, words


def map_speakers(turns, gold):
    """Optimal one-to-one system -> gold speaker assignment by overlap (Hungarian); unmatched system speakers -> X<s>.
    Equals the frozen _map_speakers when the counts match; that one maps only the first k system speakers by name when
    the system finds more speakers than the gold (W3 with an estimated count can)."""
    from scipy.optimize import linear_sum_assignment
    sys_spk, gold_spk = sorted({t["speaker"] for t in turns}), sorted({g.speaker for g in gold})
    ov = [[sum(max(0.0, min(t["end"], x.end) - max(t["start"], x.start))
               for t in turns if t["speaker"] == s for x in gold if x.speaker == g) for g in gold_spk] for s in sys_spk]
    rows, cols = linear_sum_assignment(ov, maximize=True)
    best = {sys_spk[r]: gold_spk[c] for r, c in zip(rows, cols)}
    return {s: best.get(s, f"X{s}") for s in sys_spk}


def system_version(meeting, turns_file, prefix):
    """A system segmentation (any diarization output) filled with the meeting's reference words (W3)."""
    words = reference_words(meeting)
    turns = json.loads(Path(turns_file).read_text())["turns"]
    spk = map_speakers(turns, gold_segments(meeting))
    raw = [dict(t, speaker=spk[t["speaker"]]) for t in _merge_turns(turns)]
    return _build(raw, words, _assign(words, raw), f"{meeting}.{prefix}."), words


def provenance(old, new):
    """Per new segment j: contrib (old ids), contained (old ids fully inside j), covered (all words have an old seg)."""
    words_of_old = {}
    for wi, i in old.word_seg.items():
        words_of_old.setdefault(i, set()).add(wi)
    prov = []
    for j in range(len(new.segs)):
        wj = {wi for wi, s in new.word_seg.items() if s == j}
        contrib = {old.word_seg[wi] for wi in wj if wi in old.word_seg}
        contained = {i for i in contrib if words_of_old[i] <= wj}
        prov.append({"contrib": sorted(contrib), "contained": sorted(contained),
                     "covered": all(wi in old.word_seg for wi in wj),
                     "exact_union": all(wi in old.word_seg for wi in wj) and contained == contrib})
    return prov
