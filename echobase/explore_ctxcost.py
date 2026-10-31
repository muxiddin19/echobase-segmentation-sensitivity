"""Post-hoc (deviation D6): input-token cost of the context-padded mitigation versus the frozen prompt (tokenizer only).

    PYTHONPATH=. python -m echobase.explore_ctxcost --out reports/explore_ctxcost.json
Mean prompt tokens per call over every AMI eval segment of each version and column, for the frozen prompt and for the
'ctx' variant (two neighbors on each side), built exactly as echobase.post_score builds them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from pilots.ami import meetings

from .gate import require_g1
from .llm import COLUMNS, PRIMARY, Scorer
from .post_score import CONTEXT, SplitScorer
from .score_versions import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc context cost")
    tok = AutoTokenizer.from_pretrained(PRIMARY)
    plain = Scorer.__new__(Scorer); plain.tok = tok
    ctx = SplitScorer.__new__(SplitScorer); ctx.tok = tok; ctx.variant = "ctx"
    res = {}
    for v in ("manual", "w1old", "placebo"):
        n_plain, n_ctx = [], []
        for m in meetings("eval"):
            ver = build(v, m)
            texts = [s.text for s in ver.segs]
            order = sorted(range(len(texts)), key=lambda i: (ver.segs[i].start, ver.segs[i].speaker))
            pos = {i: p for p, i in enumerate(order)}
            for i, t in enumerate(texts):
                tri = (" / ".join(texts[j] for j in order[max(0, pos[i] - CONTEXT):pos[i]]), t,
                       " / ".join(texts[j] for j in order[pos[i] + 1:pos[i] + 1 + CONTEXT]))
                for col in COLUMNS.values():
                    if col.name.startswith("doc_"):
                        continue
                    n_plain.append(len(tok(plain._prompt(col, t)).input_ids))
                    n_ctx.append(len(tok(ctx._prompt(col, tri)).input_ids))
        res[v] = {"calls": len(n_plain), "plain_tokens_mean": float(np.mean(n_plain)),
                  "ctx_tokens_mean": float(np.mean(n_ctx)), "ratio": float(np.sum(n_ctx) / np.sum(n_plain))}
        print(v, res[v], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
