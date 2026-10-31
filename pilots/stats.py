"""Locked statistics (program_pilotA.md): meeting-level bootstrap of pooled Pilot A rates."""
import random


def pooled_rates(rows):
    """rows: per-meeting dicts with counts reused, wrong, recomputed, value_changed."""
    reused = sum(r["reused"] for r in rows)
    wrong = sum(r["wrong"] for r in rows)
    recomputed = sum(r["recomputed"] for r in rows)
    changed = sum(r["value_changed"] for r in rows)
    return {
        "W": wrong / reused if reused else 0.0,
        "R": recomputed / changed if changed else float("inf"),
    }


def bootstrap(rows, n_boot=2000, seed=0, level=0.95):
    """95% percentile CIs for W and R, resampling meetings with replacement and re-pooling counts."""
    rng = random.Random(seed)
    draws = {"W": [], "R": []}
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        r = pooled_rates(sample)
        draws["W"].append(r["W"])
        draws["R"].append(r["R"])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for k, xs in draws.items():
        xs.sort()
        out[k] = (xs[int(lo_q * (len(xs) - 1))], xs[int(hi_q * (len(xs) - 1))])
    return out
