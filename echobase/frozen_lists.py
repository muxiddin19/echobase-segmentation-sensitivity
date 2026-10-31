"""E0.10 (plan v2): write configs/frozen_lists.json, the lists and settings fixed before G1.

    python -m echobase.frozen_lists
Reads only names, ids and file hashes (no eval scoring). Model revisions are the snapshot hashes in the local HF caches.
"""
import hashlib
import json
from pathlib import Path

from pilots.ami import meetings

from . import wiki
from .replay import CONF_EDGES
from .score_versions import placebo_seed

ROOT = Path(__file__).resolve().parents[1]
CACHES = [Path.home() / ".cache" / "huggingface" / "hub", Path.home() / ".cache" / "torch" / "pyannote"]
MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "pyannote/speaker-diarization-3.0", "pyannote/speaker-diarization-3.1",
          "pyannote/segmentation-3.0", "pyannote/wespeaker-voxceleb-resnet34-LM", "hbredin/wespeaker-voxceleb-resnet34-LM"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def revision(model):
    for c in CACHES:
        ref = c / f"models--{model.replace('/', '--')}" / "refs" / "main"
        if ref.exists():
            return ref.read_text().strip()
    return None


if __name__ == "__main__":
    out = {
        "ami": {s: meetings(s) for s in ("dev", "eval")},
        "ami_partition": "full-corpus-ASR (= BUTSpeechFIT/AMI-diarization-setup lists)",
        "placebo_seeds": {m: placebo_seed(m) for s in ("dev", "eval") for m in meetings(s)},
        "wiki": {"shard": wiki.SHARD, "n_articles": wiki.N_ARTICLES, "min_words": wiki.MIN_WORDS,
                 "max_words": wiki.MAX_WORDS, "seed": wiki.SEED, "window": wiki.WINDOW, "min_para": wiki.MIN_PARA,
                 "articles_sha256": sha(wiki.DATA / "articles.jsonl"),
                 "ids": {s: [a["id"] for a in wiki.articles(s)] for s in ("dev", "test")}},
        "models": {m: revision(m) for m in MODELS},
        "code_sha256": {f: sha(ROOT / f) for f in ("echobase/llm.py", "echobase/drift.py", "echobase/workloads.py",
                                                   "echobase/wiki.py", "echobase/lineage.py")},
        "settings": {"guard": 0.02, "eps_grid": [0.02, 0.05, 0.10, 0.15], "delta": 0.05, "conf_edges": list(CONF_EDGES)},
    }
    p = ROOT / "configs" / "frozen_lists.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {p}: {len(out['ami']['eval'])} eval meetings, {len(out['wiki']['ids']['test'])} test articles; "
          f"models {out['models']}")
