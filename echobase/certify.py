"""Certified reuse: audit provisional values per stratum with an exact finite-population (hypergeometric) bound.

A stratum of N provisional cells is certified when, after recomputing a simple random sample of n of them,
the (1 - alpha) upper bound on the number D of wrong provisional values satisfies D_hi / N <= eps.
Sampling follows a doubling schedule (as in sIVM); an uncertifiable stratum is recomputed in full.
Exact for sampling without replacement within a stratum; alpha = delta / (#strata x #rounds) (union bound).
"""
import math

from scipy.stats import hypergeom


def d_upper(N, n, x, alpha):
    """Largest D in [x, N - (n - x)] with P(X <= x | N, D, n) >= alpha  (exact one-sided upper bound)."""
    lo, hi = x, N - (n - x)
    while lo < hi:  # P(X <= x) is non-increasing in D: binary search for the last D that is not rejected
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(x, N, mid, n) >= alpha:
            lo = mid
        else:
            hi = mid - 1
    return lo


def audit(provisional, truth, rng, eps, alpha, n0=16):
    """Audit one stratum.

    provisional / truth: lists of equal length (truth = what a recompute returns; only sampled entries are 'read').
    Returns (final_values, llm_calls, certified, d_hi).
    """
    N = len(provisional)
    if N == 0:
        return [], 0, True, 0
    order = rng.permutation(N)
    rounds = max(1, math.ceil(math.log2(max(N / n0, 1))) + 1)
    a = alpha / rounds
    n = min(n0, N)
    while True:
        sample = order[:n]
        x = sum(provisional[i] != truth[i] for i in sample)
        dh = d_upper(N, n, x, a)
        if dh / N <= eps:
            final = list(provisional)
            for i in sample:  # sampled cells were recomputed anyway: use the recomputed value
                final[i] = truth[i]
            return final, n, True, dh
        # give up only when the stratum is confidently worse than eps (exact lower bound > eps): a raw x/n > eps
        # on a tiny sample (1 wrong in 16) abandoned strata whose true error was below eps (dev, 2026-10-28)
        if n >= N or d_lower(N, n, x, a) / N > eps:
            return list(truth), N, False, dh
        n = min(2 * n, N)


def d_lower(N, n, x, alpha):
    """Smallest D with P(X >= x | N, D, n) >= alpha  (exact one-sided lower bound)."""
    lo, hi = x, N - (n - x)
    while lo < hi:  # P(X >= x) is non-decreasing in D: binary search for the first D that is not rejected
        mid = (lo + hi) // 2
        if hypergeom.sf(x - 1, N, mid, n) >= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo
