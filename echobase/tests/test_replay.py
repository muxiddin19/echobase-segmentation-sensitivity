"""E0.2 (plan v2 §7): replay causality (policies cannot read unsampled truth) and paired same-cell comparison."""
import numpy as np

from echobase.certify import audit
from echobase.replay import paired, run_policy


def test_audit_ignores_unsampled_truth():
    gen = np.random.default_rng(1)
    for trial in range(40):
        N = int(gen.integers(20, 400))
        prov = list(gen.integers(0, 2, N))
        truth = [p if gen.random() > gen.uniform(0, 0.3) else 1 - p for p in prov]
        final, calls, ok, _ = audit(prov, truth, np.random.default_rng(trial), 0.10, 0.05)
        sampled = set(np.random.default_rng(trial).permutation(N)[:calls])  # audit samples a prefix of this order
        flipped = [1 - t if i not in sampled else t for i, t in enumerate(truth)]
        final2, calls2, ok2, _ = audit(prov, flipped, np.random.default_rng(trial), 0.10, 0.05)
        assert (calls2, ok2) == (calls, ok)  # decisions depend only on sampled truth
        if ok:  # unsampled cells keep their provisional value; sampled ones take the recompute
            assert all(final[i] == (truth[i] if i in sampled else prov[i]) for i in range(N))


def _cells(gen, n):
    rules = [None, "hash", "or-sub", "or-super"]
    return [{"col": "action_item", "change": "split", "rule": rules[i % 4], "rule_val": "no",
             "anchor_val": None if i % 7 == 0 else "yes", "truth": str(gen.choice(["yes", "no"])),
             "rule_conf": 3.0, "anchor_conf": 1.0} for i in range(n)]


def test_policies_never_read_truth_when_eps_is_one():
    # eps = 1 certifies every stratum on its first sample: call counts must not depend on truth at all
    gen = np.random.default_rng(0)
    a, b = _cells(gen, 300), _cells(gen, 300)
    for pol in ("hash", "anchor_cert", "echobase", "echobase_conf"):
        ca = run_policy(pol, a, np.random.default_rng(5), 1.0, 0.05)[0]
        cb = run_policy(pol, b, np.random.default_rng(5), 1.0, 0.05)[0]
        assert ca == cb, pol


def test_paired_uses_identical_cells():
    cs = _cells(np.random.default_rng(2), 70)
    out = paired(cs)
    eligible = [c for c in cs if c["rule"] not in (None, "hash") and c["anchor_val"] is not None]
    assert sum(r["n"] for r in out.values()) == len(eligible)
    for (_, rule), r in out.items():
        xs = [c for c in eligible if c["rule"] == rule]
        assert r["pairs"] == [(c["rule_val"], c["anchor_val"], c["truth"]) for c in xs]
        assert r["rule_err"] == sum(c["rule_val"] != c["truth"] for c in xs) / len(xs)
        assert r["anchor_err"] == sum(c["anchor_val"] != c["truth"] for c in xs) / len(xs)
