"""Post-hoc exploratory scoring (deviation D5): a larger model over both GPUs, and AMI meetings outside dev/eval.

    python -m echobase.post_score ami  --meetings eval --versions manual,w1old,placebo --model Qwen/Qwen2.5-3B-Instruct --gpus 0,1
    python -m echobase.post_score wiki --model Qwen/Qwen2.5-3B-Instruct --gpus 0,1
    python -m echobase.post_score ami  --meetings configs/ami_scale.txt --versions manual,w1old,placebo --model Qwen/Qwen2.5-0.5B-Instruct --gpus 0
    python -m echobase.post_score check --model Qwen/Qwen2.5-3B-Instruct --gpus 0,1 --out reports/post_determinism_3b.json
The prompts, answer sets and first-token scoring are the locked echobase.llm.Scorer, unchanged; only model placement
differs (device_map over the listed GPUs when there are two). Caches use the same layout as score_versions /
score_wiki under cache_root(model), so explore_eval reads them unchanged. 'check' rescores 200 eval manual segments
per column under two batch budgets and reports label mismatches.
D6 adds --variant para (reworded questions) and --variant ctx (two neighboring segments on each side as context; the
target is labeled); their caches go to cache_root(model) + "-para" / "-ctx". D7 adds rewordings para2-para4, mode
wikisize (fixed chunks of 100 and 150 words) and --cache-name (the 1.5B scale run, kept out of the v2 cache).
    python -m echobase.post_score ami --variant ctx --versions manual,w1old,placebo --model Qwen/Qwen2.5-1.5B-Instruct --gpus 0
"""
import argparse
import os
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilots.ami import meetings

from .gate import require_g1
from .llm import Scorer, cache_root
from .score_versions import build
from .wiki import articles, chunk_fixed, chunk_paragraphs, words_and_paragraphs

ROOT = Path(__file__).resolve().parents[1]
AMI_COLS = ("action_item", "question", "entity_count", "sentiment", "topic")
DOC_COLS = ("doc_date", "doc_entity_count", "doc_topic")


PARAPHRASE = {  # D6 calibration: the same questions reworded; answer sets and scoring unchanged
    "action_item": "Is anyone in this meeting utterance agreeing to, assigning or promising a task that will be done?",
    "question": "Is the speaker of this meeting utterance asking something?",
    "entity_count": "Count the different named persons, organizations or products in this meeting utterance.",
    "sentiment": "Overall, how does this meeting utterance come across?",
    "topic": "What is the main subject of this meeting utterance?",
}
PARAPHRASES = {  # D7: three further rewordings, fixed before any run; answer sets and scoring unchanged
    "para": PARAPHRASE,
    "para2": {
        "action_item": "Does someone in this meeting utterance take on, hand out or agree to a task?",
        "question": "Does this meeting utterance ask anything of the listeners?",
        "entity_count": "How many different named people, companies or products does this meeting utterance mention?",
        "sentiment": "What feeling does this meeting utterance express?",
        "topic": "This meeting utterance is mostly about which subject?",
    },
    "para3": {
        "action_item": "Would this meeting utterance belong in a list of action items (who will do what)?",
        "question": "Is there a question in this meeting utterance, direct or indirect?",
        "entity_count": "Count the named people, organisations and products that this meeting utterance refers to.",
        "sentiment": "Is the tone of this meeting utterance positive, neutral or negative?",
        "topic": "What is this meeting utterance chiefly concerned with?",
    },
    "para4": {
        "action_item": "Does this meeting utterance commit someone to doing something after the meeting?",
        "question": "Is the speaker requesting information or an answer in this meeting utterance?",
        "entity_count": "How many named entities (people, organisations, products) appear in this meeting utterance?",
        "sentiment": "How would you rate the sentiment of this meeting utterance?",
        "topic": "Which subject does this meeting utterance mainly address?",
    },
}
WINDOWS = (100, 150)  # D7: alternative fixed chunk sizes in words (the locked chunker uses 120)
CONTEXT = 2  # D6 mitigation: segments of context on each side, fixed before any run
MAX_MEM = os.environ.get("POST_MAX_MEM", "11GiB")  # weights per card; lower it to leave room for activations (Phi: 9GiB)
DTYPE = getattr(torch, os.environ.get("POST_DTYPE", "float32"))  # D6 7B on an A100: bfloat16
BUDGET = int(os.environ.get("POST_BUDGET", "3072"))  # tokens per batch (Scorer default 3072; Phi fp32 needs 1536)


class SplitScorer(Scorer):
    """Scorer whose model may span several GPUs (fp32 3B does not fit one 12 GB card), with two post-hoc prompt
    variants (D6): 'para' rewords the column question; 'ctx' shows neighboring segments as context and labels only
    the target. A text is then a (before, target, after) tuple."""

    def __init__(self, model_id, gpus, variant="none"):
        self.variant = variant
        if len(gpus) == 1:
            super().__init__(model_id=model_id, gpu=gpus[0], dtype=DTYPE)
        else:
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.tok.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=DTYPE, device_map="auto",
                max_memory={g: MAX_MEM for g in gpus}).eval()
            self.device = f"cuda:{gpus[0]}"
            self.id = f"{model_id}/{str(DTYPE).split('.')[-1]}/split"
            self.calls = 0
        if self.tok.pad_token is None:  # Llama-style tokenizers (Phi) have no pad token
            self.tok.pad_token = self.tok.eos_token

    def _first(self, word, space=True):
        """First token of the answer word, skipping a leading whitespace-only piece (SentencePiece tokenizers split
        ' 7' or '7' into a prefix-space piece and the digit). For Qwen's byte-level BPE this is the locked behavior."""
        ids = self.tok.encode((" " if space else "") + word, add_special_tokens=False)
        while len(ids) > 1 and not self.tok.decode(ids[:1]).strip():
            ids = ids[1:]
        return ids[0]

    def _prompt(self, col, text):
        if self.variant == "none":
            return super()._prompt(col, text)
        if col.kind == "binary":
            fmt = "Answer with a single word: yes or no."
        elif col.kind == "count":
            fmt = "Answer with a single digit from 0 to 9 (9 means nine or more)."
        else:
            fmt = "Answer with a single word, one of: " + ", ".join(col.options) + "."
        if self.variant in PARAPHRASES:
            user = f"{PARAPHRASES[self.variant][col.name]}\n\nUtterance: \"{text}\""
        else:
            before, target, after = text
            user = (f"{col.question}\n\nJudge only the utterance marked TARGET; the surrounding utterances are context."
                    f"\n\nBefore: \"{before}\"\nTARGET: \"{target}\"\nAfter: \"{after}\"")
        msgs = [{"role": "system", "content": fmt}, {"role": "user", "content": user}]
        suffix = "Answer: " if col.kind == "count" else "Answer:"
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + suffix


def meeting_list(spec):
    if spec in ("dev", "eval"):
        return meetings(spec)
    return [l.strip() for l in Path(spec).read_text().splitlines() if l.strip() and not l.startswith("#")]


def score_ami(scorer, out, ms, versions):
    for v in versions:
        for m in ms:
            if v == "w1old" and not (ROOT / "outputs" / "diar" / f"{m}.json").exists():
                continue
            ver = build(v, m)
            texts = [s.text for s in ver.segs]
            lens = [len(t.split()) for t in texts]
            inputs = texts
            if scorer.variant == "ctx":  # neighbors in the version's time order, any speaker
                order = sorted(range(len(texts)), key=lambda i: (ver.segs[i].start, ver.segs[i].speaker))
                pos = {i: p for p, i in enumerate(order)}
                inputs = [(" / ".join(texts[j] for j in order[max(0, pos[i] - CONTEXT):pos[i]]), texts[i],
                           " / ".join(texts[j] for j in order[pos[i] + 1:pos[i] + 1 + CONTEXT])) for i in range(len(texts))]
            for col in AMI_COLS:
                f = out / v / col / f"{m}.json"
                if f.exists():
                    continue
                vm = scorer.score(col, inputs, token_budget=BUDGET)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"values": [x for x, _ in vm], "margins": [y for _, y in vm], "lens": lens}))
            print(f"{v} {m}: {len(texts)} segments (calls so far {scorer.calls})", flush=True)


def chunk_window(words, w):
    return [(i, min(i + w, len(words))) for i in range(0, len(words), w)]


def score_wiki(scorer, out, sizes=False):
    for art in articles("test"):
        words, paras = words_and_paragraphs(art["text"])
        if sizes:  # D7: the same fixed chunker at other window sizes
            versions = [(f"fixed{w}", chunk_window(words, w)) for w in WINDOWS]
        else:
            versions = [("fixed", chunk_fixed(words)), ("para", chunk_paragraphs(words, paras))]
        for name, chunks in versions:
            texts = [" ".join(words[s:e]) for s, e in chunks]
            for col in DOC_COLS:
                f = out / "wiki" / name / col / f"{art['id']}.json"
                if f.exists():
                    continue
                vm = scorer.score(col, texts, token_budget=BUDGET)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"values": [x for x, _ in vm], "margins": [y for _, y in vm], "spans": chunks}))
    print(f"done wiki test: calls {scorer.calls}", flush=True)


def check(scorer, out, path, n=200):
    """Labels under two batch budgets; up to 200 manual segments of the first eval meetings."""
    texts = [s.text for m in meetings("eval")[:2] for s in build("manual", m).segs][:n]
    res = {}
    for col in ("action_item", "entity_count", "topic"):
        a = [v for v, _ in scorer.score(col, texts, token_budget=1024)]
        b = [v for v, _ in scorer.score(col, texts, token_budget=2 * BUDGET)]
        res[col] = {"n": len(texts), "mismatch_budgets": sum(x != y for x, y in zip(a, b))}
    Path(path).write_text(json.dumps({"model": scorer.id, "columns": res}, indent=1))
    print(res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ami", "wiki", "wikisize", "check"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--meetings", default="eval")
    ap.add_argument("--versions", default="manual,w1old,placebo")
    ap.add_argument("--variant", choices=["none", "ctx", *PARAPHRASES], default="none")  # D6/D7 prompt variants
    ap.add_argument("--cache-name", help="D7: cache dir under outputs/echobase instead of the model's (scale runs)")
    ap.add_argument("--out")
    a = ap.parse_args()
    require_g1("post-hoc scoring")
    scorer = SplitScorer(a.model, [int(g) for g in a.gpus.split(",")], a.variant)
    out = cache_root(ROOT, a.model)
    if a.variant != "none":
        out = out.with_name(out.name + f"-{a.variant}")
    if a.cache_name:
        out = ROOT / "outputs" / "echobase" / a.cache_name
    if a.mode == "ami":
        score_ami(scorer, out, meeting_list(a.meetings), a.versions.split(","))
    elif a.mode in ("wiki", "wikisize"):
        score_wiki(scorer, out, sizes=a.mode == "wikisize")
    else:
        check(scorer, out, a.out)
