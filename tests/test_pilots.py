import numpy as np
import pytest

from pilots import guarantee_eval as ge
from pilots.stats import bootstrap, pooled_rates
from pilots.verdict import verdict_pilotA, verdict_pilotB


def test_recall_lcb_matches_hand_calculation():
    # one stratum, 4 sampled units of 4 (no finite-population variance): exact recall 2/3, no width
    s = np.array([1.0, 1.0, -1.0, -1.0]); g = np.array([True, True, True, False])
    assert ge.recall_lcb(s, g, {0: 4}, {0: np.arange(4)}, 0.0) == pytest.approx(2 / 3)


def synthetic_units(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.random(n) < 0.12
    conf = rng.uniform(0.3, 1.0, n)
    noise = rng.normal(0, 1.5 * (1.3 - conf))           # low ASR confidence -> noisier scores
    score = np.where(g, 1.0, -1.0) + noise
    return [{"margin1": float(x), "margin2": -9.0, "gold1": bool(y), "gold2": False,
             "asr_conf": float(c), "empty": False} for x, y, c in zip(score, g, conf)]


@pytest.mark.parametrize("method", ["global", "lineage"])
def test_guarantee_holds_on_synthetic(method, monkeypatch):
    monkeypatch.setattr(ge, "TRIALS", 60)
    monkeypatch.setattr(ge, "BUDGETS", [800, 1600])
    res = ge.run(synthetic_units(), method, seed=0)
    for n, v in res.items():
        assert np.mean(v["viol"]) <= ge.DELTA, (method, n)        # validity of the recall guarantee
        assert all(0 <= p <= 1 for p in v["prec"])


def test_lineage_allocation_sums_to_budget():
    alloc = ge.allocate({0: 500, 1: 300, 2: 20}, {0: 0.1, 1: 0.5, 2: 1.0}, 200)
    assert sum(alloc.values()) <= 200 and all(a >= 2 for a in alloc.values())


def test_bootstrap_contains_point_estimate():
    rows = [{"reused": 10, "wrong": i % 3, "recomputed": 30, "value_changed": 10 + i} for i in range(18)]
    pt, ci = pooled_rates(rows), bootstrap(rows, n_boot=500)
    assert ci["W"][0] <= pt["W"] <= ci["W"][1] and ci["R"][0] <= pt["R"] <= ci["R"][1]


def A(**kw):
    base = {"n_meetings": 18, "W": 0.02, "R": 3.0, "W_ci95": (0.01, 0.03), "R_ci95": (2.5, 3.5), "floor": 0.0,
            "share": 0.2, "structural_counts": {"n": 1, "reused": 1, "wrong": 0, "recomputed": 1, "value_changed": 1}}
    base.update(kw)
    return base


def test_verdict_pilotA_rules():
    assert verdict_pilotA(A()) == "PASS"                                          # lo(R) >= 2
    assert verdict_pilotA(A(share=0.1)) == "FAIL"
    assert verdict_pilotA(A(R=1.1, R_ci95=(1.05, 1.2), W_ci95=(0.0, 0.04))) == "FAIL"
    assert verdict_pilotA(A(R=1.8, R_ci95=(1.5, 2.1))) == "INCONCLUSIVE"
    assert verdict_pilotA(A(n_meetings=17)) == "ANOMALY"


def test_verdict_pilotB_rules():
    budgets = [800, 1120, 1440, 1600]
    rng = np.random.default_rng(0)
    viol = {m: {n: 0.04 for n in budgets} for m in ("global", "lineage")}
    prec = {m: {n: list(rng.normal(0.30, 0.01, 300)) for n in budgets} for m in ("global", "lineage")}
    prec["lineage"][1120] = list(rng.normal(0.32, 0.01, 300))
    assert verdict_pilotB(prec, viol, 18) == "PASS"
    prec["lineage"][1120] = prec["lineage"][1440] = list(rng.normal(0.28, 0.01, 300))
    assert verdict_pilotB(prec, viol, 18) == "FAIL"
    prec["lineage"][1120] = prec["lineage"][1440] = list(rng.normal(0.30, 0.01, 300))
    assert verdict_pilotB(prec, viol, 18) == "INCONCLUSIVE"
    viol["global"][800] = 0.2
    assert verdict_pilotB(prec, viol, 18) == "ANOMALY"
