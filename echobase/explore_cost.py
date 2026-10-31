"""Post-hoc (review 4): what each workload costs to measure, next to how much it drifts (tokenizer only, CPU).

    PYTHONPATH=. python -m echobase.explore_cost --out reports/explore_cost.json
For every held-out version: scorer calls (segments x columns) and prompt tokens of the frozen prompt, built exactly as
echobase.llm.Scorer builds them for the primary model. GPU time comes from the measured rates in throughput.json where
a version was timed; for the others it is derived from the measured token rate (reported as such).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pilots.ami import meetings

from .gate import require_g1
from .lineage import system_version
from .llm import COLUMNS, PRIMARY, Scorer
from .post_score import DOC_COLS, WINDOWS, chunk_window
from .score_versions import W3_OLD, build
from .wiki import articles, chunk_fixed, chunk_paragraphs, words_and_paragraphs

ROOT = Path(__file__).resolve().parents[1]
AMI = ("manual", "w1old", "placebo", "w3old", "merge2", "merge4", "merge8")
MEET_COLS = [c for c in COLUMNS.values() if not c.name.startswith("doc_")]


def ami_texts(v, m):
    if v == "w3old":
        return [s.text for s in system_version(m, W3_OLD / f"{m}.json", "w3o")[0].segs]
    return [s.text for s in build(v, m).segs]


def wiki_versions():
    out = {"fixed": [], "para": [], **{f"fixed{w}": [] for w in WINDOWS}}
    for art in articles("test"):
        words, paras = words_and_paragraphs(art["text"])
        for name, chunks in (("fixed", chunk_fixed(words)), ("para", chunk_paragraphs(words, paras)),
                             *((f"fixed{w}", chunk_window(words, w)) for w in WINDOWS)):
            out[name] += [" ".join(words[s:e]) for s, e in chunks]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc cost per workload")
    tok = AutoTokenizer.from_pretrained(PRIMARY)
    sc = Scorer.__new__(Scorer); sc.tok = tok
    res = {}

    def count(name, texts, cols):
        n = [len(tok(sc._prompt(c, t)).input_ids) for t in texts for c in cols]
        res[name] = {"segments": len(texts), "calls": len(n), "tokens": int(np.sum(n)), "tokens_mean": float(np.mean(n))}
        print(name, res[name], flush=True)

    for v in AMI:
        count(v, [t for m in meetings("eval") for t in ami_texts(v, m)], MEET_COLS)
    for v, texts in wiki_versions().items():
        count("wiki/" + v, texts, [COLUMNS[c] for c in DOC_COLS])
    Path(a.out).write_text(json.dumps(res, indent=1))
