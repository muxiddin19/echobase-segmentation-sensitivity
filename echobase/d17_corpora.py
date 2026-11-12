"""D17-C (post-hoc; reports/deviations.md): a second corpus per domain, scored by the primary model with the frozen prompts.

Meetings: ICSI Meeting Corpus, core NXT annotations (CC BY 4.0), data/icsi/ICSI. 20 of the 75 meetings, drawn with
random.Random(crc32("icsi:d17")). Words: <w> tokens of class W with start and end times; speaker = the channel letter
of the file (words, segments and dialogue acts are stored per channel). Manual segments: the transcription segments.
Workloads: W2 placebo (echobase.workloads.placebo, seed crc32("placebo:<meeting>")) and merges k = 2, 4, 8.
Reference: question <- MRDA tags qy, qw, qr, qrr, qo, qh; action item <- cs (any component of the label).

Documents: GovReport CRS test split (US government works, public domain), data/govreport/data/crs_test.jsonl; 200
reports drawn with random.Random(crc32("crs:d17")); text = paragraphs in document order, section titles dropped.
W4 as for Wikipedia (echobase.wiki.chunk_fixed / chunk_paragraphs).

    python -m echobase.d17_corpora check --gpus 0 --out reports/d17_corpora_determinism.json   # GPU, before use
    python -m echobase.d17_corpora icsi --gpus 0                                             # GPU
    python -m echobase.d17_corpora crs --gpus 0                                              # GPU
    python -m echobase.d17_corpora analyze --out reports/d17_corpora.json                    # CPU
"""
import argparse
import bisect
import json
import random
import re
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ICSI = ROOT / "data" / "icsi" / "ICSI"
CRS = ROOT / "data" / "govreport" / "data" / "crs_test.jsonl"
PRIMARY = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL = __import__("os").environ.get("EB_MODEL", PRIMARY)  # D19: second scorer; default = primary (D17-C unchanged)
SUFFIX = {"Qwen/Qwen2.5-3B-Instruct": "-3b", "Qwen/Qwen2.5-0.5B-Instruct": "-0.5b"}.get(MODEL, "")
OUT = ROOT / "outputs" / "echobase" / f"v2-d17{SUFFIX}"
NITE = "{http://nite.sourceforge.net/}"
REF_TAGS = {"question": {"qy", "qw", "qr", "qrr", "qo", "qh"}, "action_item": {"cs"}}
ICSI_VERSIONS = ("manual", "placebo", "merge2", "merge4", "merge8")
COLS = ("action_item", "question")


# ---------------------------------------------------------------- ICSI
def icsi_all():
    w = {f.name.split(".")[0] for f in (ICSI / "Words").glob("*.words.xml")}
    d = {f.name.split(".")[0] for f in (ICSI / "DialogueActs").glob("*.dialogue-acts.xml")}
    s = {f.name.split(".")[0] for f in (ICSI / "Segments").glob("*.segs.xml")}
    return sorted(w & d & s)


def icsi_meetings():
    return sorted(random.Random(zlib.crc32(b"icsi:d17")).sample(icsi_all(), 20))


def _t(x):
    return float(x) if x not in (None, "") else None


@lru_cache(maxsize=None)
def icsi_words(m):
    out = []
    for f in sorted((ICSI / "Words").glob(f"{m}.*.words.xml")):
        ch = f.name.split(".")[1]
        for e in ET.parse(f).getroot().iter("w"):
            s, t = _t(e.get("starttime")), _t(e.get("endtime"))
            if e.get("c") == "W" and s is not None and t is not None and (e.text or "").strip():
                out.append({"word": e.text.strip(), "start": s, "end": t, "speaker": ch})
    out.sort(key=lambda w: (w["start"], w["speaker"]))
    return out


def icsi_segments(m):
    out = []
    for f in sorted((ICSI / "Segments").glob(f"{m}.*.segs.xml")):
        ch = f.name.split(".")[1]
        for e in ET.parse(f).getroot().iter("segment"):
            s, t = _t(e.get("starttime")), _t(e.get("endtime"))
            if s is not None and t is not None:
                out.append({"start": s, "end": t, "speaker": ch})
    return sorted(out, key=lambda x: (x["start"], x["speaker"]))


def icsi_acts(m):
    out = []
    for f in sorted((ICSI / "DialogueActs").glob(f"{m}.*.dialogue-acts.xml")):
        ch = f.name.split(".")[1]
        for e in ET.parse(f).getroot().iter("dialogueact"):
            s, t = _t(e.get("starttime")), _t(e.get("endtime"))
            if s is not None and t is not None:
                out.append({"start": s, "end": t, "speaker": ch,
                            "tags": set(re.split(r"[\^|.:,]", e.get("type") or ""))})
    return out


def icsi_gold(m, words, col):
    by = defaultdict(list)
    for a in icsi_acts(m):
        if a["tags"] & REF_TAGS[col]:
            by[a["speaker"]].append((a["start"], a["end"]))
    for v in by.values():
        v.sort()
    out = []
    for w in words:
        spans, mid = by.get(w["speaker"], []), (w["start"] + w["end"]) / 2
        i = bisect.bisect_right(spans, (mid, float("inf"))) - 1
        out.append(i >= 0 and spans[i][0] <= mid <= spans[i][1])
    return out


def icsi_version(v, m):
    from .lineage import _assign, _build
    from .workloads import merge_k, placebo
    words = icsi_words(m)
    raw = icsi_segments(m)
    man = _build(raw, words, _assign(words, raw), f"{m}.icsi.")
    if v == "manual":
        return man
    if v == "placebo":
        return placebo(man, words, zlib.crc32(f"placebo:{m}".encode()))
    if v.startswith("merge"):
        return merge_k(man, words, int(v[5:]))
    raise ValueError(v)


# ---------------------------------------------------------------- CRS
def _paras(sec):
    out = list(sec.get("paragraphs") or [])
    for sub in sec.get("subsections") or []:
        out += _paras(sub)
    return out


def crs_reports():
    rows = [json.loads(line) for line in CRS.open(encoding="utf-8")]
    by = {r["id"]: r for r in rows}
    pick = sorted(random.Random(zlib.crc32(b"crs:d17")).sample(sorted(by), 200))
    return [{"id": i, "text": "\n".join(p.strip() for p in _paras(by[i]["reports"]) if p.strip())} for i in pick]


# ---------------------------------------------------------------- scoring (GPU)
def score_icsi(scorer, budget):
    for v in ICSI_VERSIONS:
        for m in icsi_meetings():
            ver = icsi_version(v, m)
            texts = [s.text for s in ver.segs]
            for col in COLS:
                f = OUT / "icsi" / v / col / f"{m}.json"
                if f.exists():
                    continue
                vm = scorer.score(col, texts, token_budget=budget)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"values": [x for x, _ in vm], "margins": [y for _, y in vm],
                                         "lens": [len(t.split()) for t in texts]}))
            print(f"icsi {v} {m}: {len(texts)} segments (calls so far {scorer.calls})", flush=True)


def score_crs(scorer, budget):
    from .wiki import chunk_fixed, chunk_paragraphs, words_and_paragraphs
    for r in crs_reports():
        words, paras = words_and_paragraphs(r["text"])
        for name, chunks in (("fixed", chunk_fixed(words)), ("para", chunk_paragraphs(words, paras))):
            f = OUT / "crs" / name / "doc_date" / f"{r['id']}.json"
            if f.exists():
                continue
            vm = scorer.score("doc_date", [" ".join(words[s:e]) for s, e in chunks], token_budget=budget)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"values": [x for x, _ in vm], "margins": [y for _, y in vm], "spans": chunks}))
        print(f"crs {r['id']}: {len(words)} words (calls so far {scorer.calls})", flush=True)


def check(scorer, budget, path):
    """Determinism spot check before use: 200 ICSI manual segments and 200 CRS fixed chunks, two batch budgets."""
    from .wiki import chunk_fixed, words_and_paragraphs
    texts = [s.text for m in icsi_meetings()[:2] for s in icsi_version("manual", m).segs][:200]
    docs = []
    for r in crs_reports()[:5]:
        w, _ = words_and_paragraphs(r["text"])
        docs += [" ".join(w[s:e]) for s, e in chunk_fixed(w)]
    res = {}
    for col, tx in (("question", texts), ("action_item", texts), ("doc_date", docs[:200])):
        a = [v for v, _ in scorer.score(col, tx, token_budget=1024)]
        b = [v for v, _ in scorer.score(col, tx, token_budget=2 * budget)]
        res[col] = {"n": len(tx), "mismatch_budgets": sum(x != y for x, y in zip(a, b))}
    Path(path).write_text(json.dumps({"model": scorer.id, "columns": res}, indent=1))
    print(res, flush=True)
    return sum(c["mismatch_budgets"] for c in res.values())


# ---------------------------------------------------------------- analysis (CPU)
def analyze(out):
    from .drift import rel, share
    from .explore_oracle import compare, has_date, stat
    from .wiki import words_and_paragraphs
    res = {"design": "reports/deviations.md D17-C", "model": MODEL,
           "icsi": {"meetings": icsi_meetings()}, "crs": {}}
    pairs = {"W2": ("manual", "placebo"), "E6_k2": ("manual", "merge2"), "E6_k4": ("manual", "merge4"),
             "E6_k8": ("manual", "merge8")}
    for name, (vo, vn) in pairs.items():
        res["icsi"][name] = {}
        for col in COLS:
            D, dl, dr, ol, orf = {}, [], [], [], []
            for m in icsi_meetings():
                fo, fn = OUT / "icsi" / vo / col / f"{m}.json", OUT / "icsi" / vn / col / f"{m}.json"
                if not (fo.exists() and fn.exists()):
                    continue
                words = icsi_words(m)
                g = icsi_gold(m, words, col)
                sh = {}
                for v, f in ((vo, fo), (vn, fn)):
                    d = json.loads(f.read_text())
                    ver = icsi_version(v, m)
                    ref = [False] * len(ver.segs)
                    for wi, j in ver.word_seg.items():
                        ref[j] = ref[j] or g[wi]
                    lens = d["lens"]
                    sh[v] = (share(d["values"], lens, "yes"), share(["yes" if x else "no" for x in ref], lens, "yes"))
                D[m] = rel(sh[vo][0], sh[vn][0])
                dl.append(100 * (sh[vn][0] - sh[vo][0])); dr.append(100 * (sh[vn][1] - sh[vo][1]))
                ol.append(sh[vo][0]); orf.append(sh[vo][1])
            x = [v for v in D.values() if v is not None]
            rec = {"units": len(D), "valid": len(x)}
            if len(x) >= 3:
                s = stat(x)
                rec.update(median=s["median"], ci=s["ci"], ci_method=s["ci_method"])
            if len(dl) >= 3:
                rec["reference"] = compare(dl, dr, ol, orf)
            rec["per_unit"] = D
            res["icsi"][name][col] = rec
    D, dl, dr, ol, orf = {}, [], [], [], []
    for r in crs_reports():
        fo, fn = OUT / "crs" / "fixed" / "doc_date" / f"{r['id']}.json", OUT / "crs" / "para" / "doc_date" / f"{r['id']}.json"
        if not (fo.exists() and fn.exists()):
            continue
        w = words_and_paragraphs(r["text"])[0]
        sh = {}
        for k, f in (("fixed", fo), ("para", fn)):
            d = json.loads(f.read_text())
            lens = [e - s for s, e in d["spans"]]
            sh[k] = (share(d["values"], lens, "yes"),
                     share(["yes" if has_date(w[s:e]) else "no" for s, e in d["spans"]], lens, "yes"))
        D[r["id"]] = rel(sh["fixed"][0], sh["para"][0])
        dl.append(100 * (sh["para"][0] - sh["fixed"][0])); dr.append(100 * (sh["para"][1] - sh["fixed"][1]))
        ol.append(sh["fixed"][0]); orf.append(sh["fixed"][1])
    x = [v for v in D.values() if v is not None]
    rec = {"units": len(D), "valid": len(x)}
    if len(x) >= 3:
        s = stat(x)
        rec.update(median=s["median"], ci=s["ci"], ci_method=s["ci_method"])
    if len(dl) >= 3:
        rec["reference"] = compare(dl, dr, ol, orf)
    rec["per_unit"] = D
    res["crs"]["W4"] = rec
    # D17-C rules: H2 / H4 decision rules as a post-hoc replication, with unit minima 18 of 20 and 180 of 200
    q1 = res["icsi"]["W2"]["action_item"]
    res["rule_H2_icsi"] = bool(q1.get("valid", 0) >= 18 and q1.get("median", 0) >= 0.05 and q1["ci"][0] >= 0.02)
    w4 = res["crs"]["W4"]
    res["rule_H4_crs"] = bool(w4.get("valid", 0) >= 180 and w4.get("median", 0) >= 0.10 and w4["ci"][0] >= 0.05)
    Path(out).write_text(json.dumps(res, indent=1))
    for name in pairs:
        for col in COLS:
            r = res["icsi"][name][col]
            ref = r.get("reference", {})
            print(f"icsi {name:6s} {col:11s} D {r.get('median', float('nan')):.3f} {r.get('ci')} ({r['valid']}) "
                  f"|dL| {ref.get('abs_llm', {}).get('median', float('nan')):.1f} |dR| {ref.get('abs_ref', {}).get('median', float('nan')):.1f} "
                  f"excess {ref.get('abs_excess', {}).get('median', float('nan')):.1f} {ref.get('abs_excess', {}).get('ci')}")
    ref = w4.get("reference", {})
    print(f"crs W4 D {w4.get('median', float('nan')):.3f} {w4.get('ci')} ({w4['valid']}) |dL| {ref['abs_llm']['median']:.1f} "
          f"|dR| {ref['abs_ref']['median']:.1f} excess {ref['abs_excess']['median']:.1f} {ref['abs_excess']['ci']}")
    print("rule H2 (ICSI):", res["rule_H2_icsi"], " rule H4 (CRS):", res["rule_H4_crs"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "icsi", "crs", "analyze", "describe"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out")
    a = ap.parse_args()
    from .gate import require_g1
    require_g1("post-hoc second corpora (D17-C)")
    if a.mode == "analyze":
        analyze(a.out)
    elif a.mode == "describe":  # sizes only, no labels
        for m in icsi_meetings():
            print(m, len(icsi_words(m)), {v: len(icsi_version(v, m).segs) for v in ICSI_VERSIONS})
        r = crs_reports()
        print("crs", len(r), "reports, words median", int(np.median([len(x["text"].split()) for x in r])))
    else:
        import os
        from .post_score import SplitScorer
        budget = int(os.environ.get("POST_BUDGET", "3072"))
        sc = SplitScorer(MODEL, [int(g) for g in a.gpus.split(",")])
        if a.mode == "check":
            raise SystemExit(check(sc, budget, a.out))
        (score_icsi if a.mode == "icsi" else score_crs)(sc, budget)
