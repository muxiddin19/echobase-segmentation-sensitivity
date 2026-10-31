"""W4 document domain (plan v2): Wikipedia articles under two chunkers over the same words.

    python -m echobase.wiki prepare          # download one shard, sample 400 articles (seed 0), split 200/200
Chunker A: fixed 120-word windows. Chunker B: paragraphs, merged forward until >= 40 words.
Source: HF wikimedia/wikipedia, config 20231101.en, shard train-00000-of-00041 (CC BY-SA 3.0 / GFDL).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wiki"
SHARD = "20231101.en/train-00000-of-00041.parquet"
N_ARTICLES, MIN_WORDS, MAX_WORDS, SEED = 400, 800, 3000, 0
WINDOW, MIN_PARA = 120, 40


def prepare():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download("wikimedia/wikipedia", SHARD, repo_type="dataset", local_dir=DATA / "raw")
    t = pq.read_table(path, columns=["id", "title", "text"]).to_pylist()
    ok = [r for r in t if MIN_WORDS <= len(r["text"].split()) <= MAX_WORDS]
    rng = random.Random(SEED)
    picked = rng.sample(ok, N_ARTICLES)
    for i, r in enumerate(picked):
        r["split"] = "dev" if i < N_ARTICLES // 2 else "test"
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "articles.jsonl", "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r) + "\n")
    print(f"{len(ok)} eligible of {len(t)}; wrote {len(picked)} articles")


def articles(split):
    with open(DATA / "articles.jsonl", encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r["split"] == split]


def words_and_paragraphs(text):
    """Word list and, per paragraph, the word index range (paragraph = block separated by a blank line or newline)."""
    words, paras = [], []
    for block in text.split("\n"):
        w = block.split()
        if w:
            paras.append((len(words), len(words) + len(w)))
            words += w
    return words, paras


def chunk_fixed(words):
    return [(i, min(i + WINDOW, len(words))) for i in range(0, len(words), WINDOW)]


def chunk_paragraphs(words, paras):
    out, start = [], None
    for a, b in paras:
        start = a if start is None else start
        if b - start >= MIN_PARA:
            out.append((start, b))
            start = None
    if start is not None:  # trailing short paragraphs join the last chunk
        if out:
            out[-1] = (out[-1][0], len(words))
        else:
            out.append((start, len(words)))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        prepare()
