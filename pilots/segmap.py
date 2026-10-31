"""Locked measurement (program_pilotA.md). Do not edit during the loop.

Maps an old segmentation to a new one by temporal overlap and classifies each
connected group of segments as a change type.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    sid: str
    speaker: str
    start: float
    end: float
    text: str


def overlap(a, b):
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def link(old, new, min_overlap=0.2):
    """Edges (i, j) where old[i] and new[j] have the same speaker and overlap by at least min_overlap
    of the shorter segment. Same-speaker only: in meetings, overlapping turns from different speakers
    would otherwise chain whole stretches into one giant n:m group (smoke test, 2026-10-27)."""
    edges = []
    for i, a in enumerate(old):
        for j, b in enumerate(new):
            if a.speaker != b.speaker:
                continue
            ov = overlap(a, b)
            shorter = min(a.end - a.start, b.end - b.start)
            if shorter > 0 and ov / shorter >= min_overlap:
                edges.append((i, j))
    return edges


def components(n_old, n_new, edges):
    """Connected components of the bipartite overlap graph, as (old_ids, new_ids)."""
    parent = list(range(n_old + n_new))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(n_old + j)
    groups = {}
    for x in range(n_old + n_new):
        groups.setdefault(find(x), ([], []))
        if x < n_old:
            groups[find(x)][0].append(x)
        else:
            groups[find(x)][1].append(x - n_old)
    return [g for g in groups.values() if g[0] or g[1]]


def classify(old, new, group, time_tol=0.05):
    oi, nj = group
    if not oi:
        return "new"
    if not nj:
        return "deleted"
    if len(oi) == 1 and len(nj) == 1:
        a, b = old[oi[0]], new[nj[0]]
        same_bounds = abs(a.start - b.start) <= time_tol and abs(a.end - b.end) <= time_tol
        if a.speaker != b.speaker:
            return "relabel"
        if same_bounds and a.text == b.text:
            return "unchanged"
        return "edited"
    if len(oi) == 1:
        return "split"
    if len(nj) == 1:
        return "merge"
    return "n:m"


def change_map(old, new, min_overlap=0.2, time_tol=0.05, relabel_overlap=0.8):
    """List of (change_type, old_ids, new_ids) covering every segment exactly once.

    A segment that lost its same-speaker match (deleted) and one that gained none (new) are paired
    as a relabel when they cover nearly the same time: mutual overlap >= relabel_overlap of each.
    """
    groups = components(len(old), len(new), link(old, new, min_overlap))
    out = [(classify(old, new, g, time_tol), g[0], g[1]) for g in groups]
    deleted = [o[0] for k, o, _ in out if k == "deleted" and len(o) == 1]
    added = [n[0] for k, _, n in out if k == "new" and len(n) == 1]
    pairs, used = [], set()
    for i in deleted:
        a = old[i]
        for j in added:
            b = new[j]
            if j in used or a.speaker == b.speaker:
                continue
            ov = overlap(a, b)
            if ov >= relabel_overlap * (a.end - a.start) and ov >= relabel_overlap * (b.end - b.start):
                pairs.append((i, j))
                used.add(j)
                break
    paired_old, paired_new = {i for i, _ in pairs}, {j for _, j in pairs}
    out = [(k, o, n) for k, o, n in out
           if not (k == "deleted" and o[0] in paired_old) and not (k == "new" and n[0] in paired_new)]
    return out + [("relabel", [i], [j]) for i, j in pairs]
