"""D18 (post-hoc, exploratory; reports/deviations.md): segmentation-invariant scoring by canonical units.

    python -m echobase.d18_canonical score --gpus 0          # GPU: every distinct canonical unit of every version, once
    python -m echobase.d18_canonical analyze --out reports/d18_canonical.json   # CPU

Canonical units are derived from the words alone: per speaker, the words a version contains, in time order, cut where
the gap to the speaker's previous word is >= 0.5 s (fixed in D18 from dev gaps, no labels). Segment boundaries play no
part, so a version that keeps the same words has the same units. Labels are cached by unit text, so a unit that recurs
in another version is never scored twice.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pilots.ami import meetings, reference_words
from pilots.build_versions import normalize

from .explore_gold import GOLD, gold_words, version
from .gate import require_g1

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL = __import__("os").environ.get("EB_MODEL", PRIMARY)  # D19: second scorer; default = primary (D18 unchanged)
SUFFIX = {"Qwen/Qwen2.5-3B-Instruct": "-3b", "Qwen/Qwen2.5-0.5B-Instruct": "-0.5b"}.get(MODEL, "")
OUT = ROOT / "outputs" / "echobase" / f"v2-canon{SUFFIX}"
SEGCACHE = "v2" if MODEL == PRIMARY else "v2-" + MODEL.split("/")[-1]  # same scorer's manual-segment labels
GAP = 0.5
VERSIONS = ("manual", "w1old", "placebo", "w3old", "merge2", "merge4", "merge8")
PAIRS = {"W1": ("w1old", "manual"), "W2": ("manual", "placebo"), "W3": ("w3old", "w1old"),
         "E6_k2": ("manual", "merge2"), "E6_k4": ("manual", "merge4"), "E6_k8": ("manual", "merge8")}
COLS = ("action_item", "question")


def units(words, present):
    """Canonical units over the word indices `present`: list of (speaker, [word indices], text)."""
    by = defaultdict(list)
    for wi in sorted(present, key=lambda i: (words[i]["start"], i)):
        by[words[wi]["speaker"]].append(wi)
    out = []
    for spk, wis in sorted(by.items()):
        cur = [wis[0]]
        for a, b in zip(wis, wis[1:]):
            if words[b]["start"] - words[a]["end"] >= GAP:
                out.append(cur); cur = []
            cur.append(b)
        out.append(cur)
    return [(words[u[0]]["speaker"], u, normalize(" ".join(words[i]["word"] for i in u))) for u in out]


def version_units(v, m, words):
    return units(words, version(v, m).word_seg.keys())


def score(gpus):
    from .post_score import SplitScorer
    sc = SplitScorer(MODEL, [int(g) for g in gpus.split(",")])
    for m in meetings("eval"):
        words = reference_words(m)
        texts = sorted({t for v in VERSIONS for _, _, t in version_units(v, m, words)})
        for col in COLS:
            f = OUT / col / f"{m}.json"
            if f.exists():
                continue
            vm = sc.score(col, texts)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({t: [x, y] for t, (x, y) in zip(texts, vm)}))
        print(f"{m}: {len(texts)} distinct units (calls so far {sc.calls})", flush=True)


def prf(tp, pp, gp):
    p, r = (tp / pp if pp else 0.0), (tp / gp if gp else 0.0)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def analyze(out):
    from transformers import AutoTokenizer

    from .drift import rel
    from .llm import COLUMNS, Scorer
    from .power_sim import median_ci
    tok = AutoTokenizer.from_pretrained(MODEL)
    sc = Scorer.__new__(Scorer); sc.tok = tok
    ntok = lambda col, texts: int(sum(len(tok(sc._prompt(COLUMNS[col], t)).input_ids) for t in texts))
    res = {"design": "reports/deviations.md D18/D19", "model": MODEL, "gap_s": GAP, "drift": {}, "validity": {}, "cost": {}, "reuse": {}}
    shares = {col: {v: {} for v in VERSIONS} for col in COLS}
    val = {col: {"canonical": [0, 0, 0], "ceiling_canonical": [0, 0, 0], "segments_manual": [0, 0, 0]} for col in COLS}
    cost = {"canonical_distinct_units": 0, "per_version_segments": 0, "canonical_tokens": 0, "per_version_tokens": 0}
    reuse = {v: [0, 0] for v in VERSIONS}
    for m in meetings("eval"):
        words = reference_words(m)
        U = {v: version_units(v, m, words) for v in VERSIONS}
        man_texts = {t for _, _, t in U["manual"]}
        for v in VERSIONS:
            reuse[v][0] += sum(t in man_texts for _, _, t in U[v]); reuse[v][1] += len(U[v])
        distinct = sorted({t for v in VERSIONS for _, _, t in U[v]})
        cost["canonical_distinct_units"] += len(distinct)
        for col in COLS:
            cost["canonical_tokens"] += ntok(col, distinct)
            for v in VERSIONS:
                segs = [s.text for s in version(v, m).segs]
                cost["per_version_tokens"] += ntok(col, segs)
                if col == COLS[0]:
                    cost["per_version_segments"] += len(segs)
        for col, types in GOLD.items():
            lab = json.loads((OUT / col / f"{m}.json").read_text())
            for v in VERSIONS:
                yes = sum(len(u) for _, u, t in U[v] if lab[t][0] == "yes")
                shares[col][v][m] = yes / sum(len(u) for _, u, _ in U[v])
            g = np.array(gold_words(m, words, types))
            pred, ceil = np.zeros(len(words), bool), np.zeros(len(words), bool)
            for _, u, t in U["manual"]:
                if lab[t][0] == "yes":
                    pred[u] = True
                if g[u].any():
                    ceil[u] = True
            for k, p in (("canonical", pred), ("ceiling_canonical", ceil)):
                c = val[col][k]; c[0] += int((p & g).sum()); c[1] += int(p.sum()); c[2] += int(g.sum())
            ver = version("manual", m)
            vals = json.loads((ROOT / "outputs" / "echobase" / SEGCACHE / "manual" / col / f"{m}.json").read_text())["values"]
            ps = np.array([wi in ver.word_seg and vals[ver.word_seg[wi]] == "yes" for wi in range(len(words))])
            c = val[col]["segments_manual"]; c[0] += int((ps & g).sum()); c[1] += int(ps.sum()); c[2] += int(g.sum())
    for col in COLS:
        res["validity"][col] = {k: prf(*c) for k, c in val[col].items()}
        r = res["validity"][col]
        r["keeps_validity"] = r["canonical"]["f1"] >= 0.9 * r["segments_manual"]["f1"]  # D18 rule
        res["drift"][col] = {}
        for name, (vo, vn) in PAIRS.items():
            d = [rel(shares[col][vo][m], shares[col][vn][m]) for m in meetings("eval")]
            x = np.array([v for v in d if v is not None])
            ab = np.abs([100 * (shares[col][vn][m] - shares[col][vo][m]) for m in meetings("eval")])
            lo, hi, _ = median_ci(x, np.random.default_rng(0), 10_000)
            res["drift"][col][name] = {"valid": len(x), "median_D": float(np.median(x)), "ci": [float(lo), float(hi)],
                                       "max_D": float(x.max()), "median_abs_pts": float(np.median(ab))}
    res["reuse"] = {v: {"units": n, "reused_from_manual": k, "share": k / n} for v, (k, n) in reuse.items()}
    res["cost"] = cost
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"validity": res["validity"], "cost": cost, "reuse": res["reuse"]}, indent=1))
    for col in COLS:
        for name, r in res["drift"][col].items():
            print(col, name, f"D {r['median_D']:.4f} {r['ci']} max {r['max_D']:.4f} |d| {r['median_abs_pts']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "analyze", "describe"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out")
    a = ap.parse_args()
    require_g1("post-hoc canonical units (D18)")
    if a.mode == "score":
        score(a.gpus)
    elif a.mode == "analyze":
        analyze(a.out)
    else:  # sizes only, no labels
        for m in meetings("eval")[:4]:
            w = reference_words(m)
            U = {v: version_units(v, m, w) for v in VERSIONS}
            print(m, {v: len(u) for v, u in U.items()}, "distinct", len({t for u in U.values() for _, _, t in u}),
                  "words/unit", round(np.mean([len(x[1]) for x in U["manual"]]), 1))
