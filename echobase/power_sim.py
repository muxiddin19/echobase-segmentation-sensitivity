"""E0.9 (plan v2): power of the frozen decision rules, simulated from dev unit-level drifts.

For each workload: draw n_units (16 meetings / 200 articles) from the dev units by a smoothed bootstrap (resample, add
Gaussian noise with Silverman's bandwidth, clip at 0: new units are distinct, not copies), apply the exact rule (median
>= T and BCa 95% lower >= L; percentile interval if BCa is undefined), repeat. Writes reports/power_sim.md.
A plain resample creates ties, and BCa's jackknife then returns NaN in about a third of draws (dev check, 2026-10-28).
"""
import warnings

import numpy as np
from scipy.stats import bootstrap

from .drift import ami_drift, wiki_drift

RULES = {  # workload: (data loader, query, n_units, T, L)
    "H1 W1 diarizer->manual": (lambda: ami_drift("w1old", "manual"), "Q1", 16, 0.10, 0.05),
    "H2 W2 placebo": (lambda: ami_drift("manual", "placebo"), "Q1", 16, 0.05, 0.02),
    "H4 W4 chunker": (lambda: wiki_drift("dev"), "Qd1", 200, 0.10, 0.05),
}


def median_ci(x, rng, n_resamples=2000):
    """(low, high, method): BCa 95% CI of the median; the percentile interval when BCa is undefined (NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = bootstrap((x,), np.median, n_resamples=n_resamples, method="BCa", random_state=rng).confidence_interval
        if np.isnan(ci.low) or np.isnan(ci.high):
            ci = bootstrap((x,), np.median, n_resamples=n_resamples, method="percentile",
                           random_state=rng).confidence_interval
            return ci.low, ci.high, "percentile"
    return ci.low, ci.high, "BCa"


def rule_met(x, T, L, rng):
    """(met, method). The confirmatory rule: median >= T and 95% CI lower bound >= L."""
    if np.median(x) < T:
        return False, None
    low, _, how = median_ci(x, rng)
    return low >= L, how


def smoothed(x, n, rng):
    h = 0.9 * min(np.std(x, ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) * len(x) ** -0.2
    return np.clip(rng.choice(x, n) + rng.normal(0, h, n), 0, None)


if __name__ == "__main__":
    lines = ["# E0.9 power simulation (dev data; plan v2 decision rules)", "",
             "| hypothesis | dev units (valid) | dev median D | dev 95% CI | P(rule met) over 500 sims | percentile fallbacks |",
             "|---|---|---|---|---|---|"]
    for name, (load, q, n, T, L) in RULES.items():
        try:
            d = load()
        except FileNotFoundError as e:
            lines.append(f"| {name} | missing cache: {e.filename} | | | |")
            continue
        x = np.array([v for v in d[q].values() if v is not None])
        rng = np.random.default_rng(0)
        low, high, how = median_ci(x, rng, 10000)
        sims = [rule_met(smoothed(x, n, rng), T, L, rng) for _ in range(500)]
        hits = sum(m for m, _ in sims)
        fb = sum(h == "percentile" for _, h in sims)
        lines.append(f"| {name} (T={T}, L={L}, n={n}) | {len(x)} | {np.median(x):.3f} | "
                     f"[{low:.3f}, {high:.3f}] ({how}) | {hits / 500:.2f} | {fb} |")
    out = "\n".join(lines) + "\n"
    print(out)
    from pathlib import Path
    (Path(__file__).resolve().parents[1] / "reports" / "power_sim.md").write_text(out)
