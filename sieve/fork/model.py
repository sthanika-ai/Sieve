"""Sieve-0.8B, Sieve-2B and Sieve-2B-v2: a decision model on Qwen3.5. One state, one question, a closed set of
options in -> a probability distribution over the options out. No text generation.

Usage:
    from sieve.fork import Sieve
    s = Sieve.from_pretrained("sthanika-ai/Sieve-2B-v2")   # downloads the Qwen3.5 base once, applies the weights + head
    s.choice(state, "Which team should handle this?", ["Billing", "Technical support", "Sales"])
    -> {"decision": "Billing", "index": 0, "confidence": 0.83, "probs": {...}}

How it stays order-invariant (the property that lets an option list be shuffled with zero change
to the answer): STATE + QUESTION is encoded once into a cache (attention KV plus the Gated
DeltaNet layers' recurrent state), the cache is replicated once per option, and every option then
continues from its own identical copy, in one right-padded batch, never seeing its siblings. Each
option is therefore scored as the literal continuation "[state][question][option_i]" -- isolated
by construction, not by masking (Qwen3.5's DeltaNet layers ignore 4-D attention masks, so masking
alone cannot isolate them the way it can for a plain attention model).

This file plus a model's released weights and head are everything needed to serve it; the Qwen3.5
base weights are not re-hosted (see the model's config.json `base_model`) -- `transformers` fetches
and caches them from Qwen's own repo, same as any other adapter release.
"""
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def listed_question(question: str, options: list[str]) -> str:
    """Append every option to the question, in an order fixed by a hash of each option's own text.
    That order depends only on the SET of options (never on the order they were passed in, and
    never on their value), so a shuffled input produces this same string back -- which is why
    listing the options does not reopen the order-invariance the fork gives up nothing for."""
    ordered = sorted(options, key=lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    return question + "\n\nAll options:\n" + "\n".join(f"- {t}" for t in ordered)


# Heads (sha256 of head.safetensors) of models released and scored without the strip: Sieve-2B
UNSTRIPPED_HEADS = {"1805086a07d34c789c27500a45e28a12d8e51fcd438a9ea58e32367c794b24e3"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def choice_prefix(state: str, question: str, strip: bool = True) -> str:
    # strip=True matches the training template, which strips both fields. Sieve-2B was released and
    # scored without the strip (config.json strip_prompt, else UNSTRIPPED_HEADS).
    if strip:
        state, question = state.strip(), question.strip()
    return f"### State\n{state}\n\n### Question\n{question}\n\n### Option\n"


class _ScalarHead(torch.nn.Module):
    """One option vector -> one logit. Exactly the head the checkpoints were trained with:
    LayerNorm -> Linear(hidden, 512) -> GELU -> Dropout -> Linear(512, 1).
    Dropout is a no-op in eval() and carries no weights, which is why the state dict keys are
    net.0/net.1/net.4, not net.0/net.1/net.3."""

    def __init__(self, hidden, width=512, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden), torch.nn.Linear(hidden, width), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Sieve:
    """Loads a Qwen3.5 base model, truncates it to the read layer, applies the released LoRA
    adapter and scalar head, and serves order-invariant Choice decisions."""

    def __init__(self, model_id, read_layer, lora_state, head_state, list_options, T=1.0,
                 lora_config=None, device=None, dtype=torch.float32, max_len=2048, full_state=None, strip=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.T, self.list_options, self.max_len, self.strip = T, list_options, max_len, strip
        self.base_model = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
        self.model = lm.model.to(self.device).eval()  # text tower only; the checkpoint also carries a vision tower
        del lm
        for p in self.model.parameters():
            p.requires_grad = False
        self.config = self.model.config
        self.hidden = self.config.hidden_size
        self.read_layer = read_layer
        self.model.layers = self.model.layers[:read_layer]  # nothing past the read layer ever runs

        if full_state is not None:  # full fine-tune: the released weights replace the base model's blocks 0..read_layer-1
            names = {n for n, _ in self.model.named_parameters()}
            missing = set(full_state) - names
            assert not missing, f"full weights not found in the model: {sorted(missing)[:3]}"
            self.model.load_state_dict(full_state, strict=False)

        if lora_config is not None:
            from peft import LoraConfig, inject_adapter_in_model
            lc = LoraConfig(r=lora_config["r"], lora_alpha=lora_config["lora_alpha"],
                            lora_dropout=0.0, target_modules=lora_config["target_modules"])
            # the adapter name must match training's (weight keys are named ...lora_A.<name>.weight)
            inject_adapter_in_model(lc, self.model, adapter_name=lora_config.get("adapter_name", "default"))
            missing = set(lora_state) - {n for n, _ in self.model.named_parameters()}
            assert not missing, f"adapter weight names not found in the model: {sorted(missing)[:3]}"
            self.model.load_state_dict(lora_state, strict=False)

        self._captured, self._recording = {}, False
        self._schemas = {}  # name -> {"a", "b", "n"}, filled in by fit_schema (optional, label-based)
        self.model.layers[-1].register_forward_hook(self._grab)  # only the read layer's output is ever needed

        self.head = _ScalarHead(self.hidden).to(self.device).eval()
        self.head.load_state_dict(head_state)

    def _grab(self, _mod, _inp, out):
        if self._recording:
            self._captured["h"] = out[0] if isinstance(out, tuple) else out

    # ---------------------------------------------------------------- tokenization
    def _ids(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def _trim(self, pre, opts):
        over = len(pre) + max(len(o) for o in opts) - self.max_len
        return pre[over:] if over > 0 else pre

    # ---------------------------------------------------------------- the fork
    def _prefix_pass(self, prefixes):
        B, Pmax = len(prefixes), max(len(p) for p in prefixes)
        ids = torch.full((B, Pmax), self.pad_id, dtype=torch.long)
        att = torch.zeros(B, Pmax, dtype=torch.long)
        pos = torch.zeros(B, Pmax, dtype=torch.long)
        for b, p in enumerate(prefixes):  # LEFT-padded: a DeltaNet pad token's state update is
            n = len(p)                    # then exactly 0 (see modeling_qwen3_5.py's gated delta rule)
            ids[b, Pmax - n:] = torch.tensor(p)
            att[b, Pmax - n:] = 1
            pos[b, Pmax - n:] = torch.arange(n)
        cache = DynamicCache(config=self.config)
        self._recording = False
        with torch.no_grad():
            out = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                             position_ids=pos.to(self.device), past_key_values=cache, use_cache=True)
        return out.past_key_values, att.to(self.device)

    def _fork_pass(self, cache, pre_att, prefix_lens, opt_rows, row_src):
        R, Omax = len(opt_rows), max(len(o) for o in opt_rows)
        src = torch.tensor(row_src, device=self.device)
        cache.reorder_cache(src)
        ids = torch.full((R, Omax), self.pad_id, dtype=torch.long)
        att = torch.zeros(R, Omax, dtype=torch.long)
        pos = torch.zeros(R, Omax, dtype=torch.long)
        for r, (o, s) in enumerate(zip(opt_rows, row_src)):  # RIGHT-padded: pads sit after the pooled token
            n, P = len(o), prefix_lens[s]
            ids[r, :n] = torch.tensor(o)
            att[r, :n] = 1
            pos[r, :n] = torch.arange(P, P + n)
            pos[r, n:] = P + n
        full_att = torch.cat([pre_att.index_select(0, src), att.to(self.device)], dim=1)
        self._captured.clear()
        self._recording = True
        try:
            with torch.no_grad():
                out = self.model(input_ids=ids.to(self.device), attention_mask=full_att,
                                 position_ids=pos.to(self.device), past_key_values=cache, use_cache=True)
            h = self._captured["h"].float()
        finally:
            self._recording = False
            self._captured.clear()
        last = torch.tensor([len(o) - 1 for o in opt_rows], device=self.device)
        return h[torch.arange(R, device=self.device), last]  # [R, H]

    @torch.no_grad()
    def _probs(self, state, question, options):
        q = listed_question(question, options) if self.list_options else question
        o_ids = [self._ids(o) for o in options]
        pre_ids = self._trim(self._ids(choice_prefix(state, q, self.strip)), o_ids)
        cache, pre_att = self._prefix_pass([pre_ids])
        h = self._fork_pass(cache, pre_att, [len(pre_ids)], o_ids, [0] * len(o_ids))
        logits = self.head(h) / self.T
        return F.softmax(logits, dim=-1).cpu().numpy()

    # ---------------------------------------------------------------- public API
    def choice(self, state, question, options, schema=None):
        """Order-invariant Choice decision: options may be passed in any order with no change to
        the result. Returns the decision, its calibrated confidence, and the full distribution.

        Confidence is calibrated two ways, and you can use either or both:
        - out of the box, with no labels: a single temperature fit once on data this model never
          trained or was tested on (`self.T`, in config.json), applied automatically here;
        - optionally sharper, if you have some labels for YOUR schema: call `fit_schema(name,
          examples)` once with ~100 labelled (state, question, options, gold_index) examples, then
          pass `schema=name` here. This is a strict upgrade over the built-in temperature for that
          specific schema, and never required."""
        p = self._probs(state, question, options)
        i = int(p.argmax())
        conf = float(p[i])
        cfg = self._schemas.get(schema) if schema else None
        if cfg is not None:
            z = np.log(np.clip(conf, 1e-6, 1 - 1e-6) / (1 - np.clip(conf, 1e-6, 1 - 1e-6)))
            conf = float(1 / (1 + np.exp(-(cfg["a"] * z + cfg["b"]))))
        return {"decision": options[i], "index": i, "confidence": conf,
                "probs": {o: float(x) for o, x in zip(options, p)}}

    def fit_schema(self, name, examples):
        """Optional, label-based upgrade over the built-in temperature, for one specific kind of
        question ("schema"): a 2-parameter logistic (Platt) refit of the top-1 confidence, fit on
        ~100 labelled examples `[(state, question, options, gold_index), ...]`. Never changes which
        option is picked (the argmax), only how trustworthy its confidence number is. Call once per
        schema, then pass `schema=name` to `choice(...)`."""
        from sklearn.linear_model import LogisticRegression
        conf, correct = [], []
        for state, question, options, gold in examples:
            p = self._probs(state, question, options)
            conf.append(float(p.max()))
            correct.append(float(int(p.argmax()) == gold))
        conf, correct = np.array(conf), np.array(correct)
        if len(np.unique(correct)) < 2:
            return None  # degenerate sample (e.g. every example correct): keep the built-in temperature
        z = np.log(np.clip(conf, 1e-6, 1 - 1e-6) / (1 - np.clip(conf, 1e-6, 1 - 1e-6)))
        lr = LogisticRegression(C=1.0).fit(z[:, None], correct)
        cfg = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "n": len(examples), "accuracy": float(correct.mean())}
        self._schemas[name] = cfg
        return cfg

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, repo_id, device=None, dtype=torch.float32, revision=None):
        from huggingface_hub import snapshot_download
        path = repo_id if os.path.isdir(repo_id) else snapshot_download(repo_id, revision=revision,
                                                                       allow_patterns=["config.json", "*.safetensors"])
        cfg = json.load(open(os.path.join(path, "config.json")))
        full_path = os.path.join(path, "full_weights.safetensors")
        full_state = load_file(full_path) if os.path.exists(full_path) else None
        lora_state = None if full_state is not None else load_file(os.path.join(path, "adapter_model.safetensors"))
        head_path = os.path.join(path, "head.safetensors")
        head_state = load_file(head_path)  # keys already net.0/net.1/net.4.*
        strip = cfg.get("strip_prompt")
        if strip is None:
            strip = _sha256(head_path) not in UNSTRIPPED_HEADS
        return cls(cfg["base_model"], cfg["read_layer"], lora_state, head_state, cfg["list_options"],
                   T=cfg.get("temperature", 1.0), lora_config=None if full_state is not None else cfg["lora"], device=device, dtype=dtype,
                   max_len=cfg.get("max_len", 2048), full_state=full_state, strip=strip)
