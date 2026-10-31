from pilots.segmap import Segment, change_map
from pilots.metrics import baseline_decisions, evaluate, self_disagreement


def S(sid, spk, s, e, t):
    return Segment(sid, spk, s, e, t)


def kinds(old, new):
    return sorted(k for k, _, _ in change_map(old, new))


def test_unchanged_and_edited():
    old = [S("a", "A", 0, 2, "hello"), S("b", "B", 3, 5, "yes")]
    new = [S("a", "A", 0, 2, "hello"), S("b", "B", 3, 5, "yes please")]
    assert kinds(old, new) == ["edited", "unchanged"]


def test_split_merge_relabel_new_deleted():
    old = [S("a", "A", 0, 4, "one two"), S("b", "B", 5, 6, "x"), S("c", "B", 6.1, 7, "y"),
           S("d", "C", 8, 9, "z"), S("e", "D", 20, 21, "gone")]
    new = [S("1", "A", 0, 2, "one"), S("2", "A", 2, 4, "two"), S("3", "B", 5, 7, "x y"),
           S("4", "E", 8, 9, "z"), S("5", "F", 30, 31, "new")]
    assert kinds(old, new) == ["deleted", "merge", "new", "relabel", "split"]


def test_cross_speaker_overlap_does_not_chain():
    # A and B talk over each other; each speaker's segment is unchanged, so nothing may merge across speakers
    old = [S("a", "A", 0, 5, "x"), S("b", "B", 4, 9, "y"), S("c", "A", 8, 12, "z")]
    new = [S("a", "A", 0, 5, "x"), S("b", "B", 4, 9, "y"), S("c", "A", 8, 12, "z")]
    assert kinds(old, new) == ["unchanged", "unchanged", "unchanged"]


def test_partial_overlap_other_speaker_is_not_relabel():
    old = [S("a", "A", 0, 4, "x")]
    new = [S("b", "B", 2, 6, "x")]
    assert kinds(old, new) == ["deleted", "new"]


def test_every_segment_covered_once():
    old = [S(str(i), "A", i, i + 1.5, "t") for i in range(5)]
    new = [S(str(i), "A", i * 1.3, i * 1.3 + 1, "t") for i in range(5)]
    groups = change_map(old, new)
    assert sorted(i for _, o, _ in groups for i in o) == list(range(5))
    assert sorted(j for _, _, n in groups for j in n) == list(range(5))


def test_baseline_wrong_reuse_and_over_recompute():
    old = [S("a", "A", 0, 2, "hi"), S("b", "A", 3, 5, "do it")]
    new = [S("a", "A", 0, 2, "hi"), S("b", "A", 3, 5, "do it now")]
    dec = baseline_decisions(old, new)
    assert dec[0] == (0, True) and dec[1] == (1, False)
    # value of reused cell differs under recompute -> wrong reuse; recomputed cell unchanged -> over-recompute
    r = evaluate(old, new, old_values=["no", "yes"], new_values=["yes", "yes"], decisions=dec)
    assert r["wrong_reuse_rate"] == 1.0
    assert r["recomputed"] == 1 and r["value_changed"] == 1


def test_self_disagreement():
    assert self_disagreement(["y", "n", "y", "y"], ["y", "y", "y", "y"]) == 0.25


def test_baseline_reuses_identical_text_despite_moved_bounds():
    old = [S("a", "A", 0, 2, "lets do it")]
    new = [S("a", "A", 0.3, 2.4, "lets do it")]
    assert baseline_decisions(old, new)[0] == (0, True)
