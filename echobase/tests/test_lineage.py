import random

from pilots.build_versions import _map_speakers
from pilots.segmap import Segment
from echobase.lineage import map_speakers


def _data(rng, n_sys, n_gold):
    gold = [Segment(str(i), "ABCD"[i % n_gold], i * 10.0, i * 10.0 + rng.uniform(3, 12), "") for i in range(40)]
    turns = [{"start": g.start + rng.uniform(-1, 1), "end": g.end + rng.uniform(-1, 1),
              "speaker": f"S{('ABCD'.index(g.speaker) * 7 + (i % 2) * (n_sys > n_gold)) % n_sys}"}
             for i, g in enumerate(gold)]
    return turns, gold


def test_equals_frozen_mapping_when_counts_match():
    rng = random.Random(0)
    for _ in range(20):
        turns, gold = _data(rng, 4, 4)
        assert map_speakers(turns, gold) == _map_speakers(turns, gold)


def test_extra_system_speakers_get_the_best_subset():
    # S9 sorts first by name but overlaps nobody; the frozen mapping would give it a gold label
    gold = [Segment("0", "A", 0, 10, ""), Segment("1", "B", 10, 20, "")]
    turns = [{"start": 0, "end": 10, "speaker": "S1"}, {"start": 10, "end": 20, "speaker": "S2"},
             {"start": 30, "end": 31, "speaker": "S0"}]
    assert map_speakers(turns, gold) == {"S0": "XS0", "S1": "A", "S2": "B"}
