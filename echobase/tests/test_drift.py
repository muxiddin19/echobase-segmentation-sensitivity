from echobase.drift import dist, rel, share


def test_share_is_word_weighted_and_segment_count_invariant():
    # same words labelled the same way, cut into 1 vs 3 segments -> identical share
    assert share(["yes"], [6], "yes") == share(["yes", "yes", "yes"], [2, 2, 2], "yes") == 1.0
    assert share(["yes", "no"], [1, 3], "yes") == 0.25


def test_rel_guard_and_value():
    assert rel(0.01, 0.5) is None
    assert abs(rel(0.20, 0.25) - 0.25) < 1e-12


def test_dist_sums_to_one():
    d = dist(["a", "b", "a"], [1, 2, 1])
    assert abs(sum(d.values()) - 1) < 1e-12 and d["a"] == 0.5
