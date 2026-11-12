"""D17-A addendum (post-hoc, descriptive; reports/deviations.md): the word-level F1 ceiling of segment-level labels and
segment-level agreement of every scorer with the reference labeling. CPU only.

    python -m echobase.d17_ceiling --out reports/d17_ceiling.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pilots.ami import meetings, reference_words

from .explore_gold import GOLD, gold_words, version
from .gate import require_g1
from .llm import cache_root
from .explore_oracle import MODELS

ROOT = Path(__file__).resolve().parents[1]
VERS = ("manual", "w1old", "placebo")
CACHES = {m.split("/")[-1]: cache_root(ROOT, m) for m in MODELS}
CACHES["learned (D17-A)"] = ROOT / "outputs" / "echobase" / "v2-learned"


def prf(tp, pp, gp):
    p, r = (tp / pp if pp else 0.0), (tp / gp if gp else 0.0)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main(out):
    require_g1("post-hoc ceiling (D17-A addendum)")
    res = {"design": "reports/deviations.md D17-A addendum", "ceiling_word_f1": {}, "segment_level": {}}
    for col, types in GOLD.items():
        ceil = {v: [0, 0, 0] for v in VERS}
        seg = {}
        for m in meetings("eval"):
            words = reference_words(m)
            g = np.array(gold_words(m, words, types))
            for v in VERS:
                ver = version(v, m)
                ref = [False] * len(ver.segs)
                for wi, j in ver.word_seg.items():
                    ref[j] = ref[j] or bool(g[wi])
                pred = np.array([wi in ver.word_seg and ref[ver.word_seg[wi]] for wi in range(len(words))])
                c = ceil[v]
                c[0] += int((pred & g).sum()); c[1] += int(pred.sum()); c[2] += int(g.sum())
                for name, V in CACHES.items():
                    f = V / v / col / f"{m}.json"
                    if not f.exists():
                        continue
                    vals = json.loads(f.read_text())["values"]
                    if len(vals) != len(ref):
                        raise ValueError(f"{name} {v} {col} {m}: {len(vals)} labels, {len(ref)} segments")
                    yes = np.array([x == "yes" for x in vals]); r = np.array(ref)
                    s = seg.setdefault(name, {}).setdefault(v, [0, 0, 0, 0])
                    s[0] += int((yes & r).sum()); s[1] += int(yes.sum()); s[2] += int(r.sum()); s[3] += 1
        res["ceiling_word_f1"][col] = {v: prf(*c) for v, c in ceil.items()}
        res["segment_level"][col] = {n: {v: dict(prf(*s[:3]), meetings=s[3]) for v, s in d.items()} for n, d in seg.items()}
    Path(out).write_text(json.dumps(res, indent=1))
    for col in GOLD:
        print(col, "ceiling word F1", {v: round(x["f1"], 3) for v, x in res["ceiling_word_f1"][col].items()})
        for n, d in res["segment_level"][col].items():
            print(f"  {n:24s}", {v: (round(x['f1'], 3), x["meetings"]) for v, x in d.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
