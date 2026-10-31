from collections import Counter

from pilots.segmap import Segment
from echobase.lineage import Version
from echobase.workloads import merge_k, placebo


def toy():
    words = [{"start": float(i), "end": i + 0.5, "word": f"w{i}", "speaker": "A" if i < 10 else "B"} for i in range(16)]
    # A: segments of 2, 3, 5 words; B: 4, 2
    cuts = {"A": [2, 3, 5], "B": [4, 2]}
    segs, ws, pos = [], {}, 0
    for spk, lens in cuts.items():
        for n in lens:
            idx = list(range(pos, pos + n))
            for wi in idx:
                ws[wi] = len(segs)
            segs.append(Segment(str(len(segs)), spk, float(idx[0]), idx[-1] + 0.5, " ".join(f"w{i}" for i in idx)))
            pos += n
    return Version(segs, ws), words


def lens_by_speaker(v):
    c = {}
    for s in v.segs:
        c.setdefault(s.speaker, Counter())[len(s.text.split())] += 1
    return c


def test_placebo_preserves_words_counts_and_lengths():
    v, words = toy()
    p = placebo(v, words, seed=3)
    assert sorted(p.word_seg) == sorted(v.word_seg)                       # same words
    assert len(p.segs) == len(v.segs)                                      # same segment count
    assert lens_by_speaker(p) == lens_by_speaker(v)                        # same length multiset per speaker
    order = [wi for s in sorted(set(p.word_seg.values())) for wi in sorted(w for w, j in p.word_seg.items() if j == s)]
    assert sorted(order) == list(range(16))


def test_merge_k():
    v, words = toy()
    m = merge_k(v, words, 2)
    assert [len(s.text.split()) for s in m.segs if s.speaker == "A"] == [5, 5]
    assert [len(s.text.split()) for s in m.segs if s.speaker == "B"] == [6]
