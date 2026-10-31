"""Locked verdict rules, transcribed from program_pilotA.md / program_pilotB.md (do not edit after G1)."""

import numpy as np

DELTA = 0.1


def verdict_pilotA(r):
    """PASS / FAIL / INCONCLUSIVE / ANOMALY for the structural condition (structural subset)."""
    W, R, (W_lo, W_hi), (R_lo, R_hi), floor = r["W"], r["R"], r["W_ci95"], r["R_ci95"], r["floor"]
    if r["n_meetings"] < 18 or W == 1.0 or R < 1.0 or min(r["structural_counts"].values()) < 0:
        return "ANOMALY"
    if r["share"] < 0.15:
        return "FAIL"
    if W_lo > max(3 * floor, 0.05) or R_lo >= 2:
        return "PASS"
    if W_hi <= max(1.5 * floor, 0.05) and R_hi < 1.3:
        return "FAIL"
    return "INCONCLUSIVE"


def _diff_ci(a, b, n_boot=2000, seed=0):
    """95% bootstrap CI of mean(a) - mean(b), resampling trials independently in each arm."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def verdict_pilotB(prec, viol, n_meetings):
    """prec[method][n]: per-trial precisions pooled over seeds; viol[method][n]: violation share."""
    budgets = sorted(prec["lineage"])
    mean = {m: {n: float(np.mean(v)) for n, v in prec[m].items()} for m in prec}
    # 13 = dev meetings with AMI dialogue-act annotations (deviation D1, reports/deviations.md)
    if (n_meetings < 13 or any(viol["global"][n] > DELTA for n in budgets)
            or any(p == 1.0 for m in mean.values() for p in m.values())):
        return "ANOMALY"
    pass_lo, _ = _diff_ci(prec["lineage"][1120], prec["global"][1600])
    _, fail_hi = _diff_ci(prec["lineage"][1440], prec["global"][1600])
    if all(viol["lineage"][n] == 0 for n in budgets) and mean["lineage"][800] >= mean["global"][1600]:
        return "ANOMALY"  # label saving > 50% with zero violations: suspiciously good
    if any(viol["lineage"][n] > DELTA for n in budgets) or fail_hi < 0:
        return "FAIL"
    if pass_lo >= 0:
        return "PASS"
    return "INCONCLUSIVE"
