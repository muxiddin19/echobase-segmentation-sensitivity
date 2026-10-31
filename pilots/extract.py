"""LLM extraction column: a yes/no question per segment, scored by one forward pass (no sampling).

The answer is "yes" iff logit(" yes") > logit(" no") for the first answer token, so reruns are
deterministic up to GPU numerics; the margin is kept for analysis.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = {
    "action_item": "Does this meeting utterance assign or commit to an action item (someone will do something)?",
    "decision": "Does this meeting utterance state a decision that the group has made?",
    # Pilot B predicates (gold: AMI dialogue-act classes sug / el.inf + el.ass)
    "suggestion": "Is this meeting utterance a suggestion (proposing an idea or a course of action)?",
    "elicit": "Does this meeting utterance ask someone for information or for their opinion or assessment?",
}


class Extractor:
    def __init__(self, model_id="Qwen/Qwen2.5-3B-Instruct", gpu=0, dtype=torch.float16, revision="main"):
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
        self.model.to(f"cuda:{gpu}").eval()
        self.device = f"cuda:{gpu}"
        self.yes = self.tok.encode(" yes", add_special_tokens=False)[0]
        self.no = self.tok.encode(" no", add_special_tokens=False)[0]
        self.id = f"{model_id}@{revision}/{str(dtype).split('.')[-1]}"

    def _prompt(self, question, text):
        msgs = [{"role": "system", "content": "Answer with a single word: yes or no."},
                {"role": "user", "content": f"{question}\n\nUtterance: \"{text}\""}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "Answer:"

    @torch.no_grad()
    def __call__(self, column, texts, token_budget=3072, max_batch=32):
        """Returns [(label, margin)] per text, label in {"yes", "no"}.

        Prompts are sorted by length and batched under a padded-token budget, so one long segment
        cannot blow up a batch on a 12 GB card; results come back in input order.
        """
        prompts = [self._prompt(QUESTIONS[column], t) for t in texts]
        lengths = [len(self.tok(p).input_ids) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])
        out = [None] * len(prompts)
        start = 0
        while start < len(order):
            end = start + 1
            while (end < len(order) and end - start < max_batch
                   and lengths[order[end]] * (end - start + 1) <= token_budget):
                end += 1
            idx = order[start:end]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            # only the last position is needed; full-sequence logits over a 152k vocabulary do not fit in 12 GB
            hidden = self.model.model(**enc).last_hidden_state[:, -1, :]
            logits = self.model.lm_head(hidden).float()
            for i, m in zip(idx, (logits[:, self.yes] - logits[:, self.no]).tolist()):
                out[i] = ("yes" if m > 0 else "no", m)
            start = end
        return out
