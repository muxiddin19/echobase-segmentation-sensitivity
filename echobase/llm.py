"""LLM columns scored by one forward pass over first-token logits (deterministic; no sampling).

Column kinds (EchoBase decomposability classes):
  binary  -> "yes"/"no"          OR-decomposable candidates (existential predicates)
  count   -> 0..9 (9 = 9 or more) SUM-decomposable candidates
  label   -> one of a fixed set   holistic (no algebraic rule; reuse must be certified)
"""
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Column:
    name: str
    kind: str          # binary | count | label
    question: str
    options: tuple = ()


COLUMNS = {c.name: c for c in [
    Column("action_item", "binary",
           "Does this meeting utterance assign or commit to an action item (someone will do something)?"),
    Column("question", "binary", "Does this meeting utterance contain a question?"),
    Column("entity_count", "count",
           "How many distinct named people, organisations or products are mentioned in this meeting utterance?"),
    Column("sentiment", "label", "What is the overall sentiment of this meeting utterance?",
           ("positive", "neutral", "negative")),
    Column("topic", "label", "Which topic is this meeting utterance mainly about?",
           ("design", "cost", "planning", "technical", "other")),
    # W4 document domain (passages of Wikipedia articles)
    Column("doc_date", "binary", "Does this passage mention a specific year or date?"),
    Column("doc_entity_count", "count",
           "How many distinct named people, organisations or places are mentioned in this passage?"),
    Column("doc_topic", "label", "Which topic is this passage mainly about?",
           ("science", "history", "arts", "sports", "politics", "geography", "other")),
]}


PRIMARY = "Qwen/Qwen2.5-1.5B-Instruct"
SECOND = "Qwen/Qwen2.5-0.5B-Instruct"  # admitted at E0.7 (SmolLM2-1.7B failed the yes-rate band)


def cache_root(root, model_id):
    """Cache directory for one model: the primary model keeps the v2 layout, other models get v2-<name>."""
    return root / "outputs" / "echobase" / ("v2" if model_id == PRIMARY else "v2-" + model_id.split("/")[-1])


class Scorer:
    def __init__(self, model_id=PRIMARY, gpu=0, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(f"cuda:{gpu}").eval()
        self.device = f"cuda:{gpu}"
        self.id = f"{model_id}/{str(dtype).split('.')[-1]}"
        self.calls = 0  # LLM invocations (one per scored text), the cost unit of the paper

    def _first(self, word, space=True):
        return self.tok.encode((" " if space else "") + word, add_special_tokens=False)[0]

    def _targets(self, col):
        if col.kind == "binary":
            words = ["yes", "no"]
        elif col.kind == "count":
            words = [str(i) for i in range(10)]
        else:
            words = list(col.options)
        # Qwen splits " 7" into " " + "7"; count prompts end with "Answer: " and score the bare digit tokens
        ids = [self._first(w, space=col.kind != "count") for w in words]
        assert len(set(ids)) == len(ids), f"first tokens collide for {col.name}: {words}"
        return words, ids

    def _prompt(self, col, text):
        if col.kind == "binary":
            fmt = "Answer with a single word: yes or no."
        elif col.kind == "count":
            fmt = "Answer with a single digit from 0 to 9 (9 means nine or more)."
        else:
            fmt = "Answer with a single word, one of: " + ", ".join(col.options) + "."
        msgs = [{"role": "system", "content": fmt},
                {"role": "user", "content": f"{col.question}\n\nUtterance: \"{text}\""}]
        suffix = "Answer: " if col.kind == "count" else "Answer:"
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + suffix

    def __call__(self, column, texts, **kw):
        """Values in input order: 'yes'/'no' (binary), int (count) or an option string (label)."""
        return [v for v, _ in self.score(column, texts, **kw)]

    @torch.no_grad()
    def score(self, column, texts, token_budget=3072, max_batch=32):
        """(value, margin) in input order; margin = top-1 minus top-2 logit over the answer tokens (confidence)."""
        col = COLUMNS[column]
        words, ids = self._targets(col)
        prompts = [self._prompt(col, t) for t in texts]
        lengths = [len(self.tok(p).input_ids) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])
        out, start = [None] * len(prompts), 0
        while start < len(order):
            end = start + 1
            while (end < len(order) and end - start < max_batch
                   and lengths[order[end]] * (end - start + 1) <= token_budget):
                end += 1
            idx = order[start:end]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            hidden = self.model.model(**enc).last_hidden_state[:, -1, :]
            logits = self.model.lm_head(hidden).float()[:, ids]
            top2 = logits.topk(2, dim=1)
            for i, k, (l1, l2) in zip(idx, top2.indices[:, 0].tolist(), top2.values.tolist()):
                out[i] = (int(words[k]) if col.kind == "count" else words[k], l1 - l2)
            self.calls += len(idx)
            start = end
        return out
