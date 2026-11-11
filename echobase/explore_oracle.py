"""D14 (post-hoc, exploratory; reports/deviations.md): non-LLM reference scorers and a series-level check. CPU only.

    python -m echobase.explore_oracle --out reports/explore_oracle.json

A word-weighted share of an "any part" predicate changes with segmentation even under a scorer that never errs on
its own terms (a unit-of-aggregation effect). We apply deterministic reference scorers to exactly the segments the
LLMs scored and split each unit's LLM change dL into the reference change dR and the remainder dL - dR:
- documents: a rule-based date detector on each cached chunk span (fixed, para, fixed100, fixed150);
- meetings: a segment is "yes" if any of its words lies in a human dialogue act of the column's type
  (explore_gold's word projection: question <- elicit acts, action item <- suggestions).
Plus, for H1 and H2, per-series and leave-one-series-out medians (the 16 meetings are 4 series x 4 sessions).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from pilots.ami import meetings

from .explore_gold import GOLD, gold_words, version
from .gate import require_g1
from .lineage import versions
from .llm import PRIMARY, cache_root
from .power_sim import median_ci
from .wiki import articles, words_and_paragraphs

ROOT = Path(__file__).resolve().parents[1]
MODELS = (PRIMARY, "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "Qwen/Qwen2.5-14B-Instruct", "microsoft/Phi-3.5-mini-instruct",
          "Qwen/Qwen2.5-32B-Instruct")  # D15: included where its caches are complete
AMI_PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo"), "W3": ("w3old", "w1old"),
             "E6_k2": ("manual", "merge2"), "E6_k4": ("manual", "merge4"), "E6_k8": ("manual", "merge8")}
WIKI_PAIRS = {"W4": ("fixed", "para"), "120->150": ("fixed", "fixed150"), "120->100": ("fixed", "fixed100")}
GUARD = 0.02

MONTHS = r"January|February|March|April|June|July|August|September|October|November|December"
YEAR = re.compile(r"^(1\d{3}|20\d{2})s?[.,;:)\]'\"]*$")
MONTH = re.compile(rf"^\(?({MONTHS})[.,;:]?$")
NUM = re.compile(r"^\(?\d{1,4}[.,;:)]*$")


def has_date(words):
    """Rule-based: a standalone year 1000-2099 (optionally a decade 's'), a month name, or 'May' next to a number."""
    for i, w in enumerate(words):
        t = w.strip("(\"'[")
        if YEAR.match(t) or MONTH.match(t):
            return True
        if t.rstrip(".,;:") == "May" and ((i > 0 and NUM.match(words[i - 1])) or
                                           (i + 1 < len(words) and NUM.match(words[i + 1]))):
            return True
    return False


def share(vals, lens):
    tot = sum(lens)
    return sum(n for v, n in zip(vals, lens) if v) / tot if tot else 0.0


def stat(x):
    x = np.asarray(x, float)
    lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
    return {"n": len(x), "median": float(np.median(x)), "ci": [float(lo), float(hi)], "ci_method": how}


def compare(dl, dr, old_l, old_r):
    """Per-unit LLM and reference changes (points) -> summary."""
    dl, dr = np.array(dl), np.array(dr)
    rel_r = [abs(d) / o for d, o in zip(dr / 100, old_r) if o >= GUARD]
    rel_l = [abs(d) / o for d, o in zip(dl / 100, old_l) if o >= GUARD]
    rho = spearmanr(dl, dr).correlation if np.std(dl) > 0 and np.std(dr) > 0 else None
    out = {"abs_llm": stat(np.abs(dl)), "abs_ref": stat(np.abs(dr)), "abs_excess": stat(np.abs(dl - dr)),
           "signed_llm": stat(dl), "signed_ref": stat(dr), "spearman": None if rho is None else float(rho),
           "rel_llm_median": float(np.median(rel_l)) if rel_l else None,
           "rel_ref_median": float(np.median(rel_r)) if rel_r else None, "rel_ref_valid": len(rel_r)}
    out["llm_adds"] = out["abs_excess"]["ci"][0] > 1.0                     # D14 rule 1
    out["definition_explains_most"] = out["abs_ref"]["median"] >= 0.5 * out["abs_llm"]["median"]  # D14 rule 2
    return out


def wiki_part():
    arts = articles("test")
    ref = {}                                           # chunker -> article -> (share, spans)
    words_of = {a["id"]: words_and_paragraphs(a["text"])[0] for a in arts}
    res = {}
    for name, (co, cn) in WIKI_PAIRS.items():
        res[name] = {}
        for model in MODELS:
            V = cache_root(ROOT, model) / "wiki"
            if not (V / cn / "doc_date").exists():
                continue
            dl, dr, ol, orf = [], [], [], []
            for a in arts:
                fo, fn = V / co / "doc_date" / f"{a['id']}.json", V / cn / "doc_date" / f"{a['id']}.json"
                if not (fo.exists() and fn.exists()):
                    continue
                w = words_of[a["id"]]
                sh = {}
                for ch, f in ((co, fo), (cn, fn)):
                    d = json.loads(f.read_text())
                    lens = [e - s for s, e in d["spans"]]
                    rv = [has_date(w[s:e]) for s, e in d["spans"]]
                    sh[ch] = (share([v == "yes" for v in d["values"]], lens), share(rv, lens))
                ol.append(sh[co][0]); orf.append(sh[co][1])
                dl.append(100 * (sh[cn][0] - sh[co][0])); dr.append(100 * (sh[cn][1] - sh[co][1]))
            if dl:
                res[name][model.split("/")[-1]] = compare(dl, dr, ol, orf)
    return res


def ami_part():
    res = {}
    for name in AMI_PAIRS:
        res[name] = {}
    for col, types in GOLD.items():
        for name, (vo, vn) in AMI_PAIRS.items():
            per_model = {}
            for m in meetings("eval"):
                words = versions(m)[2]
                g = gold_words(m, words, types)
                ref_sh, ok = {}, True
                for v in (vo, vn):
                    ver = version(v, m)
                    seg_yes, seg_len = {}, {}
                    for wi, j in ver.word_seg.items():
                        seg_len[j] = seg_len.get(j, 0) + 1
                        seg_yes[j] = seg_yes.get(j, False) or g[wi]
                    ref_sh[v] = share([seg_yes[j] for j in seg_len], [seg_len[j] for j in seg_len])
                for model in MODELS:
                    V = cache_root(ROOT, model)
                    fo, fn = V / vo / col / f"{m}.json", V / vn / col / f"{m}.json"
                    if not (fo.exists() and fn.exists()):
                        continue
                    do, dn = json.loads(fo.read_text()), json.loads(fn.read_text())
                    so = share([v == "yes" for v in do["values"]], do["lens"])
                    sn = share([v == "yes" for v in dn["values"]], dn["lens"])
                    r = per_model.setdefault(model.split("/")[-1], ([], [], [], []))
                    r[0].append(100 * (sn - so)); r[1].append(100 * (ref_sh[vn] - ref_sh[vo]))
                    r[2].append(so); r[3].append(ref_sh[vo])
            for mk, (dl, dr, ol, orf) in per_model.items():
                if len(dl) == 16:
                    res[name].setdefault(col, {})[mk] = compare(dl, dr, ol, orf)
    return res


def series_part():
    conf = {h["id"]: h for h in json.loads((ROOT / "reports" / "confirmatory.json").read_text())}
    out = {}
    for hid, T in (("H1", 0.10), ("H2", 0.05)):
        pu = conf[hid]["per_unit"]
        series = sorted({m[:-1] for m in pu})
        per = {s: float(np.median([v for m, v in pu.items() if m[:-1] == s])) for s in series}
        loso = {s: float(np.median([v for m, v in pu.items() if m[:-1] != s])) for s in series}
        out[hid] = {"T": T, "series_median": per, "leave_one_series_out_median": loso,
                    "robust": all(v >= T for v in loso.values()),
                    "breaks_without": [s for s, v in loso.items() if v < T]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc reference scorers (D14)")
    res = {"design": "reports/deviations.md D14", "wiki": wiki_part(), "ami": ami_part(), "series": series_part()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, r in res["wiki"].items():
        for mk, c in r.items():
            print(f"wiki {name:9s} {mk:24s} |dL| {c['abs_llm']['median']:5.1f} |dR| {c['abs_ref']['median']:5.1f} "
                  f"|dL-dR| {c['abs_excess']['median']:5.1f} {c['abs_excess']['ci']} rho {c['spearman']}")
    for name, cols in res["ami"].items():
        for col, r in cols.items():
            for mk, c in r.items():
                print(f"ami  {name:6s} {col:11s} {mk:24s} |dL| {c['abs_llm']['median']:5.1f} "
                      f"|dR| {c['abs_ref']['median']:5.1f} |dL-dR| {c['abs_excess']['median']:5.1f} "
                      f"[{c['abs_excess']['ci'][0]:.1f}, {c['abs_excess']['ci'][1]:.1f}]")
    print(json.dumps(res["series"], indent=1))
