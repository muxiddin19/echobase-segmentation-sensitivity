"""Score both chunkings of the Wikipedia articles (W4) with the document columns; cache (value, margin, len).

    python -m echobase.score_wiki --split dev --gpu 1
Test-split scoring is refused before G1 (plan v2).
"""
import argparse
import json
from pathlib import Path

from .gate import require_g1
from .llm import PRIMARY, Scorer, cache_root
from .wiki import articles, chunk_fixed, chunk_paragraphs, words_and_paragraphs

ROOT = Path(__file__).resolve().parents[1]
COLS = ("doc_date", "doc_entity_count", "doc_topic")

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "test"])
ap.add_argument("--gpu", type=int, default=1)
ap.add_argument("--model", default=PRIMARY)  # E4: the admitted second model
a = ap.parse_args()
if a.split == "test":
    require_g1("test scoring")
scorer = Scorer(model_id=a.model, gpu=a.gpu)
OUT = cache_root(ROOT, a.model) / "wiki"
for art in articles(a.split):
    words, paras = words_and_paragraphs(art["text"])
    for name, chunks in (("fixed", chunk_fixed(words)), ("para", chunk_paragraphs(words, paras))):
        texts = [" ".join(words[s:e]) for s, e in chunks]
        for col in COLS:
            f = OUT / name / col / f"{art['id']}.json"
            if f.exists():
                continue
            vm = scorer.score(col, texts)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"values": [x for x, _ in vm], "margins": [y for _, y in vm],
                                     "spans": chunks}))
print(f"done {a.split}: calls {scorer.calls}", flush=True)
