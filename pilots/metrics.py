"""Locked measurement (program_pilotA.md). Do not edit during the loop.

Compares the strong baseline's reuse decisions with full recompute.
"""
from .segmap import overlap

STRUCTURAL = {"split", "merge", "n:m"}


def baseline_decisions(old, new, time_tol=0.05):
    """Time-overlap anchoring + dependency invalidation (reuse iff the anchor's text is unchanged).

    Returns {new_index: anchor_old_index or None, "reuse": bool} per new segment.
    """
    out = {}
    for j, b in enumerate(new):
        # anchor: the most-overlapping old segment of the same speaker; any speaker only if none (relabel)
        best, best_ov = None, 0.0
        for same in (True, False):
            for i, a in enumerate(old):
                if (a.speaker == b.speaker) != same:
                    continue
                ov = overlap(a, b)
                if ov > best_ov:
                    best, best_ov = i, ov
            if best is not None:
                break
        if best is None:
            out[j] = (None, False)
            continue
        # the extraction reads only the text, so a strong baseline reuses whenever the input text is
        # identical (content-hash cache), even if boundaries moved; time_tol is kept for the signature
        out[j] = (best, old[best].text == b.text)
    return out


def evaluate(old, new, old_values, new_values, decisions, new_ids=None):
    """Wrong-reuse rate and over-recompute ratio, restricted to new_ids if given.

    old_values[i] / new_values[j]: extraction output on old segment i / new segment j
    (new_values = full recompute, the reference).
    """
    ids = range(len(new)) if new_ids is None else new_ids
    reused = wrong = recomputed = changed = 0
    for j in ids:
        anchor, reuse = decisions[j]
        value_changed = anchor is None or old_values[anchor] != new_values[j]
        changed += value_changed
        if reuse:
            reused += 1
            wrong += value_changed
        else:
            recomputed += 1
    return {
        "n": len(list(ids)),
        "wrong_reuse_rate": wrong / reused if reused else 0.0,
        "over_recompute_ratio": recomputed / changed if changed else float("inf") if recomputed else 1.0,
        "reused": reused, "wrong": wrong, "recomputed": recomputed, "value_changed": changed,
    }


def self_disagreement(values_run1, values_run2):
    """Noise floor: share of cells where two runs of the same extraction disagree."""
    pairs = list(zip(values_run1, values_run2))
    return sum(a != b for a, b in pairs) / len(pairs) if pairs else 0.0
