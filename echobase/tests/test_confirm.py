"""Preregistered verdict logic (echobase/confirm.py) on hand-made unit drifts."""
import numpy as np

from echobase.confirm import report, run, verdict


def rng():
    return np.random.default_rng(0)


def test_supported_refuted_inconclusive():
    hi = {i: 0.2 + 0.01 * i for i in range(16)}
    assert verdict(hi, 0.10, 0.05, 14, rng())["verdict"] == "supported"
    lo = {i: 0.01 + 0.001 * i for i in range(16)}
    assert verdict(lo, 0.10, 0.05, 14, rng())["verdict"] == "refuted"
    mixed = {i: v for i, v in enumerate([0.0] * 8 + [0.3] * 8)}
    assert verdict(mixed, 0.10, 0.05, 14, rng())["verdict"] == "inconclusive"


def test_guard_and_missing_units_count_against_minimum():
    d = {i: (None if i < 2 else 0.3 + 0.01 * i) for i in range(15)}  # 2 guard-excluded, 1 of 16 missing -> 13 valid
    r = verdict(d, 0.10, 0.05, 14, rng(), n_units=16)
    assert (r["missing"], r["excluded_guard"], r["valid"], r["verdict"]) == (1, 2, 13, "inconclusive")


def test_fixed_sequence_stops_claims_at_first_failure():
    good = lambda: ({"q": {i: 0.3 + 0.01 * i for i in range(16)}}, list(range(16)))
    bad = lambda: ({"q": {i: 0.01 for i in range(16)}}, list(range(16)))
    res = run([("A", "", (good, "q"), 0.1, 0.05, 14), ("B", "", (bad, "q"), 0.1, 0.05, 14),
               ("C", "", (good, "q"), 0.1, 0.05, 14)])
    assert [r["claimed"] for r in res] == [True, False, False]
    assert [r["verdict"] for r in res] == ["supported", "refuted", "supported"]
    assert "| C |" in report(res)
