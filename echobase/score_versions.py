"""Score one segmentation version of the AMI meetings with all columns; cache (value, margin) per segment.

    python -m echobase.score_versions --version placebo --split dev --gpu 0
Versions: manual (AMI manual), w1old (pyannote 3.1 + reference words), w3old (pyannote 3.1, estimated speaker count + reference words; D3),
placebo (W2), merge2/4/8 (E6).
Eval-split scoring is refused before G1 (plan v2: never touch eval before the freeze).
"""
import argparse
import json
import zlib
from pathlib import Path

from pilots.ami import meetings

from .gate import require_g1
from .lineage import system_version, versions
from .llm import PRIMARY, Scorer, cache_root
from .workloads import merge_k, placebo

ROOT = Path(__file__).resolve().parents[1]
W3_OLD = ROOT / "outputs" / "diar_v" / "3.1-est"  # W3 old side, exploratory (D3; D2 threshold variant changed 0.4%)
LOCK = ROOT / ".claude" / "eval.lock.json"


def placebo_seed(meeting):
    return zlib.crc32(f"placebo:{meeting}".encode())  # fixed per meeting; recorded in configs/frozen_lists.json


def build(version, meeting):
    old, new, words = versions(meeting)
    if version == "manual":
        return new
    if version == "w1old":
        return old
    if version == "placebo":
        return placebo(new, words, placebo_seed(meeting))
    if version.startswith("merge"):
        return merge_k(new, words, int(version[5:]))
    if version == "w3old":  # 3.1 estimated-count turns + reference words (W3 old side; new side = w1old, 3.1 default)
        return system_version(meeting, W3_OLD / f"{meeting}.json", "w3o")[0]
    raise ValueError(version)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--columns", default="action_item,question,entity_count,sentiment,topic")  # AMI columns only
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=PRIMARY)  # E4: the admitted second model
    a = ap.parse_args()
    if a.split == "eval":
        require_g1("eval scoring")
    scorer = Scorer(model_id=a.model, gpu=a.gpu)
    out = cache_root(ROOT, a.model)
    for m in meetings(a.split):
        if a.version == "w3old" and not (W3_OLD / f"{m}.json").exists():
            continue
        v = build(a.version, m)
        texts = [s.text for s in v.segs]
        lens = [len(t.split()) for t in texts]
        for col in a.columns.split(","):
            f = out / a.version / col / f"{m}.json"
            if f.exists():
                continue
            vm = scorer.score(col, texts)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"values": [x for x, _ in vm], "margins": [y for _, y in vm], "lens": lens}))
        print(f"{a.version} {m}: {len(texts)} segments (calls so far {scorer.calls})", flush=True)
