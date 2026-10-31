"""Re-segmentation workloads over a fixed word sequence (plan v2 §4.2).

W2 placebo: per speaker, keep the words (in time order), the number of segments and the multiset of segment word
lengths; lay the lengths out in a random order. Granularity is matched, only boundary placement changes.
E6 granularity: merge k consecutive same-speaker segments.
"""
import random

from pilots.segmap import Segment

from .lineage import Version


def _speaker_words(version, words):
    """speaker -> time-ordered word indices of that speaker's segments, plus their segment lengths."""
    by_seg = {}
    for wi, s in version.word_seg.items():
        by_seg.setdefault(s, []).append(wi)
    out = {}
    for s, wis in sorted(by_seg.items(), key=lambda kv: version.segs[kv[0]].start):
        spk = version.segs[s].speaker
        ws, lens = out.setdefault(spk, ([], []))
        wis = sorted(wis, key=lambda wi: words[wi]["start"])
        ws.extend(wis)
        lens.append(len(wis))
    return out


def _cut(spk, wis, lens, words, prefix, start_id):
    segs, word_seg, pos = [], {}, 0
    for k, n in enumerate(lens):
        piece = wis[pos:pos + n]
        pos += n
        sid = start_id + k
        text = " ".join(words[wi]["word"] for wi in piece)
        segs.append((Segment(f"{prefix}{sid}", spk, words[piece[0]]["start"], words[piece[-1]]["end"], text), piece))
    return segs


def _assemble(pieces):
    pieces.sort(key=lambda p: (p[0].start, p[0].speaker))
    segs, word_seg = [], {}
    for j, (seg, piece) in enumerate(pieces):
        segs.append(seg)
        for wi in piece:
            word_seg[wi] = j
    return segs, word_seg


def placebo(version, words, seed, prefix="placebo."):
    from pilots.build_versions import normalize
    rng = random.Random(seed)
    pieces = []
    for spk, (wis, lens) in _speaker_words(version, words).items():
        lens = list(lens)
        rng.shuffle(lens)
        pieces += _cut(spk, wis, lens, words, prefix, len(pieces))
    segs, word_seg = _assemble(pieces)
    segs = [Segment(s.sid, s.speaker, s.start, s.end, normalize(s.text)) for s in segs]
    return Version(segs, word_seg)


def merge_k(version, words, k, prefix="merge."):
    from pilots.build_versions import normalize
    pieces = []
    for spk, (wis, lens) in _speaker_words(version, words).items():
        merged = [sum(lens[i:i + k]) for i in range(0, len(lens), k)]
        pieces += _cut(spk, wis, merged, words, prefix + f"k{k}.", len(pieces))
    segs, word_seg = _assemble(pieces)
    segs = [Segment(s.sid, s.speaker, s.start, s.end, normalize(s.text)) for s in segs]
    return Version(segs, word_seg)
