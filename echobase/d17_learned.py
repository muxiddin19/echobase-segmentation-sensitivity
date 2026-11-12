"""D17-A (post-hoc, exploratory; reports/deviations.md): a high-validity, non-LLM learned scorer. CPU only.

    python -m echobase.d17_learned --out reports/d17_learned_fit.json

Trains a TF-IDF + logistic-regression segment classifier per column (question, action item) on AMI manual segments of
training meetings (every annotated meeting outside the dev and eval series), labels a segment "yes" if any of its words
lies in a dialogue act of the column's types (the D14 reference definition), chooses C and the probability threshold on
the 18 dev meetings by word-level F1, and then labels every eval version once. The labels are written as a scorer cache
(outputs/echobase/v2-learned, same format as the LLM caches), so the released statistics code reads them unchanged:
    python -m echobase.explore_eval --partial --cache v2-learned --out reports/d17_learned_eval.json
    python -m echobase.explore_signed --cache v2-learned --out reports/d17_learned_signed.json
    python -m echobase.explore_gold --cache v2-learned --out reports/d17_learned_gold.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pilots.ami import ANN, gold_segments, meetings, reference_words

from .explore_gold import GOLD, gold_words
from .gate import require_g1
from .lineage import _assign, _build
from .score_versions import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "echobase" / "v2-learned"
VERSIONS = ("manual", "w1old", "placebo", "w3old", "merge2", "merge4", "merge8")
HELD_SERIES = ("ES2011", "IB4", "IS1008", "TS3004", "EN2002", "ES2004", "IS1009", "TS3003")  # dev and eval series
CS = (0.1, 1.0, 10.0)
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 ... 0.90


def train_meetings():
    have = sorted({f.name.split(".")[0] for f in (ANN / "dialogueActs").glob("*.dialog-act.xml")})
    held = set(meetings("dev")) | set(meetings("eval"))
    return [m for m in have if m not in held and not m.startswith(HELD_SERIES)]


def manual(m):
    """AMI manual segments over reference words, without any diarization file (as lineage.versions' new side)."""
    words = reference_words(m)
    raw = [{"start": g.start, "end": g.end, "speaker": g.speaker} for g in gold_segments(m)]
    return _build(raw, words, _assign(words, raw), f"{m}.gold."), words


def seg_labels(ver, gw):
    yes = [False] * len(ver.segs)
    for wi, j in ver.word_seg.items():
        yes[j] = yes[j] or gw[wi]
    return yes


def word_f1(ver, words, gw, seg_pred):
    pred = np.array([wi in ver.word_seg and seg_pred[ver.word_seg[wi]] for wi in range(len(words))])
    g = np.array(gw)
    tp = int((pred & g).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / g.sum() if g.sum() else 0.0
    return tp, int(pred.sum()), int(g.sum()), (2 * p * r / (p + r) if p + r else 0.0)


def pooled_f1(rows):
    tp, pp, gp = (sum(r[i] for r in rows) for i in range(3))
    p, r = (tp / pp if pp else 0.0), (tp / gp if gp else 0.0)
    return 2 * p * r / (p + r) if p + r else 0.0


def main(out):
    require_g1("post-hoc learned scorer (D17-A)")
    tr = train_meetings()
    data = {}
    for m in tr:
        try:
            ver, words = manual(m)
        except Exception as e:  # meeting without words or segments in the release
            print("skip", m, type(e).__name__)
            continue
        data[m] = (ver, words)
    dev = {m: build("manual", m) for m in meetings("dev")}
    dev_words = {m: reference_words(m) for m in dev}
    report = {"design": "reports/deviations.md D17-A", "train_meetings": sorted(data), "columns": {}}
    for col, types in GOLD.items():
        X, y = [], []
        for m, (ver, words) in data.items():
            lab = seg_labels(ver, gold_words(m, words, types))
            X += [s.text for s in ver.segs]
            y += lab
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xt = vec.fit_transform(X)
        dev_gw = {m: gold_words(m, dev_words[m], types) for m in dev}
        dev_X = {m: vec.transform([s.text for s in dev[m].segs]) for m in dev}
        best, grid = None, []
        for C in CS:
            clf = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=1000).fit(Xt, y)
            probs = {m: clf.predict_proba(dev_X[m])[:, 1] for m in dev}
            for t in THRESHOLDS:
                f1 = pooled_f1([word_f1(dev[m], dev_words[m], dev_gw[m], probs[m] >= t) for m in dev])
                grid.append({"C": C, "threshold": t, "dev_word_f1": f1})
                if best is None or f1 > best[2]:
                    best = (C, t, f1, clf)
        C, t, f1, clf = best
        report["columns"][col] = {"n_train_segments": len(y), "train_yes_share": float(np.mean(y)),
                                  "chosen": {"C": C, "threshold": t, "dev_word_f1": f1}, "grid": grid}
        print(f"{col}: {len(y)} train segments, yes {np.mean(y):.3f}; chosen C={C} t={t} dev F1 {f1:.3f}", flush=True)
        for v in VERSIONS:  # eval, once
            for m in meetings("eval"):
                ver = build(v, m)
                texts = [s.text for s in ver.segs]
                p = clf.predict_proba(vec.transform(texts))[:, 1]
                f = OUT / v / col / f"{m}.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"values": ["yes" if x >= t else "no" for x in p],
                                         "margins": [float(abs(x - t)) for x in p],
                                         "lens": [len(s.split()) for s in texts]}))
    Path(out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
