"""Segmentation drift of query answers (plan v2 §3.2, §4.3): D(u) = |Q(new) - Q(old)| / Q(old), Q(old) >= 0.02.

AMI: old/new are two segmentations of the same reference words; queries are word-weighted shares (segment-count
invariant) except SUM(entity_count) and topic TVD. Wikipedia: chunk 'fixed' (old) vs 'para' (new).
"""
import json
from collections import Counter
from pathlib import Path

from pilots.ami import meetings

from .lineage import versions

ROOT = Path(__file__).resolve().parents[1]
V1, V2 = ROOT / "outputs" / "echobase" / "values", ROOT / "outputs" / "echobase" / "v2"
GUARD = 0.02

AMI_QUERIES = {"Q1": ("action_item", "yes"), "Q2": ("question", "yes"), "Q3": ("sentiment", "negative"),
               "Q4": ("topic", "cost")}
WIKI_QUERIES = {"Qd1": ("doc_date", "yes")}


def ami_load(version, col, m, cache={}):
    """(values, lengths) of one AMI version: the v2 cache (score_versions; all eval versions) when it exists, else for
    manual / w1old on dev the exploratory caches."""
    f = V2 / version / col / f"{m}.json"
    if version in ("manual", "w1old") and not f.exists():
        if m not in cache:
            old, new, _ = versions(m)
            cache[m] = {"w1old": [len(s.text.split()) for s in old.segs],
                        "manual": [len(s.text.split()) for s in new.segs]}
        tag = "new" if version == "manual" else "old"
        return json.loads((V1 / col / f"{m}.{tag}.json").read_text()), cache[m][version]
    d = json.loads(f.read_text())
    return d["values"], d["lens"]


def share(vals, lens, target):
    return sum(n for v, n in zip(vals, lens) if v == target) / sum(lens)


def dist(vals, lens):
    c = Counter()
    for v, n in zip(vals, lens):
        c[v] += n
    t = sum(c.values())
    return {k: x / t for k, x in c.items()}


def rel(old, new):
    return None if old < GUARD else abs(new - old) / old


def ami_drift(old_version, new_version, split="dev", only=None):
    """{query: {meeting: D}} plus SUM-entities relative change and topic TVD; `only` restricts to some meetings."""
    out = {q: {} for q in list(AMI_QUERIES) + ["Q5_entities", "Q6_topic_tvd"]}
    for m in (only if only is not None else meetings(split)):
        for q, (col, target) in AMI_QUERIES.items():
            vo, lo = ami_load(old_version, col, m)
            vn, ln = ami_load(new_version, col, m)
            out[q][m] = rel(share(vo, lo, target), share(vn, ln, target))
        eo, _ = ami_load(old_version, "entity_count", m)
        en, _ = ami_load(new_version, "entity_count", m)
        out["Q5_entities"][m] = abs(sum(en) - sum(eo)) / max(sum(eo), 1)
        to, lo = ami_load(old_version, "topic", m)
        tn, ln = ami_load(new_version, "topic", m)
        do, dn = dist(to, lo), dist(tn, ln)
        out["Q6_topic_tvd"][m] = 0.5 * sum(abs(do.get(k, 0) - dn.get(k, 0)) for k in set(do) | set(dn))
    return out


def wiki_drift(split="dev", only=None):
    from .wiki import articles
    out = {"Qd1": {}, "Qd2_entities": {}, "Qd3_topic_tvd": {}}
    for art in articles(split):
        if only is not None and art["id"] not in only:
            continue
        def load(chunker, col):
            d = json.loads((V2 / "wiki" / chunker / col / f"{art['id']}.json").read_text())
            return d["values"], [e - s for s, e in d["spans"]]
        vo, lo = load("fixed", "doc_date")
        vn, ln = load("para", "doc_date")
        out["Qd1"][art["id"]] = rel(share(vo, lo, "yes"), share(vn, ln, "yes"))
        eo, _ = load("fixed", "doc_entity_count")
        en, _ = load("para", "doc_entity_count")
        out["Qd2_entities"][art["id"]] = abs(sum(en) - sum(eo)) / max(sum(eo), 1)
        to, lo = load("fixed", "doc_topic")
        tn, ln = load("para", "doc_topic")
        do, dn = dist(to, lo), dist(tn, ln)
        out["Qd3_topic_tvd"][art["id"]] = 0.5 * sum(abs(do.get(k, 0) - dn.get(k, 0)) for k in set(do) | set(dn))
    return out
