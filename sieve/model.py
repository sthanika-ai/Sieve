"""Sieve model: a Qwen backbone without its language-model head, and a pointer head over the options."""
import copy
import math
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .calibration import question_temperatures
from .encode import encode, rows_of


class PointerHead(nn.Module):
    """logit_i = <k(h_</opt>_i), q(h_<decide>)> / sqrt(dp); divided by the calibration temperature in eval mode."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1.0 / math.sqrt(dp)
        self.temperature = 1.0

    def forward(self, h_decide, h_opts, temperature=None):
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        t = self.temperature if temperature is None else temperature
        return z if (self.training or t == 1.0) else z / t


class SieveModel(nn.Module):
    # A single question with a branch at least this long runs as one causal row (state + branch) instead of
    # continuing from the cached state, so attention stays on the fused causal kernel.
    RECOMPUTE_MIN_BRANCH = int(os.environ.get("SIEVE_RECOMPUTE_MIN_BRANCH", "1024"))

    def __init__(self, name, device="cuda", dtype=torch.bfloat16, attn="sdpa", dp=256, head_dtype=torch.float32):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(name)
        full = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=attn, device_map={"": device})
        self.lm = full.model
        for attr in ("language_model", "text_model"):
            if hasattr(self.lm, attr):
                self.lm = getattr(self.lm, attr)
                break
        self.lm.eval()
        cfg = self.lm.config
        self.d = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        self.head = PointerHead(self.d, dp).to(device=device, dtype=head_dtype).eval()
        self.device, self.dtype, self.head_dtype = device, dtype, head_dtype
        self.pad_id = self.tok.pad_token_id or 0
        self.temperatures = None        # per-category temperatures from head.pt (None: the one global temperature)
        self.max_state = self.max_branch = 8192    # token limits; load_sieve sets the ones the model was trained with

    def question_temperatures(self, meta, state=None):
        """One calibration temperature per question (meta from api.to_record): its category's temperature from
        head.pt, or the global one. None when head.pt has no per-category table (every question uses the global T)."""
        return question_temperatures(self.temperatures, meta, self.head.temperature, state)

    def encode(self, state, questions, **kw):
        kw.setdefault("max_state", self.max_state)
        kw.setdefault("max_branch", self.max_branch)
        return encode(self.tok, state, questions, **kw)

    def load_head(self, path):
        d = torch.load(os.path.join(path, "head.pt"), map_location=self.device, weights_only=False)
        self.head.load_state_dict(d["head"])
        self.head.temperature = d.get("temperature", 1.0)
        self.temperatures = d.get("temperatures")      # per question category (Sieve-9B-Plus); None: the global T
        return self

    def _new_cache(self):
        if "linear_attention" in set(getattr(self.lm.config, "layer_types", None) or []):
            return DynamicCache(config=self.lm.config)
        return DynamicCache()

    @staticmethod
    def _cache_layers(cache):
        if hasattr(cache, "key_cache"):
            return list(zip(cache.key_cache, cache.value_cache))
        if hasattr(cache, "layers"):
            return [(l.keys, l.values) for l in cache.layers]
        raise RuntimeError(f"unrecognised KV cache layout: {type(cache).__name__}")

    @classmethod
    def _replicate(cls, cache, Q):
        """Copy a batch-1 cache to Q rows, leaving the original intact so it can be reused."""
        if any(hasattr(l, "recurrent_states") for l in getattr(cache, "layers", []) or []):
            new = copy.deepcopy(cache)
            new.reorder_cache(torch.zeros(Q, dtype=torch.long))
            return new
        new = DynamicCache()
        for i, (k, v) in enumerate(cls._cache_layers(cache)):
            new.update(k.expand(Q, -1, -1, -1).contiguous(), v.expand(Q, -1, -1, -1).contiguous(), i)
        return new

    @torch.no_grad()
    def prefill_from_record(self, rec):
        """Run the state once; the result can be reused by every request with the same state."""
        Ls = rec["n_state"]
        t = torch.tensor([rec["ids"][:Ls]], device=self.device)
        out = self.lm(input_ids=t, past_key_values=self._new_cache(), use_cache=True)
        return {"n": Ls, "cache": out.past_key_values,
                "last_hidden": out.last_hidden_state[0, -1].to(self.head_dtype)}

    @torch.no_grad()
    def logits(self, rec, prefix=None, temperature=None):
        """One logit vector per question. Questions continue from the (cached) state as a batch of causal rows.

        temperature: None (the stored global one), a number, or a list with one per question."""
        Ls = rec["n_state"]
        _, _, rows = rows_of(rec)
        tq = temperature if isinstance(temperature, (list, tuple)) else [temperature] * len(rows)
        if (self.RECOMPUTE_MIN_BRANCH and len(rows) == 1 and len(rows[0]["ids"]) >= self.RECOMPUTE_MIN_BRANCH
                and Ls <= len(rows[0]["ids"])):
            r = rows[0]
            ids = torch.tensor([rec["ids"][:Ls] + r["ids"]], device=self.device)
            h = self.lm(input_ids=ids).last_hidden_state[0].to(self.head_dtype)
            idx = torch.tensor([Ls + o for o in r["opts"]], device=self.device)
            return [self.head(h[Ls + r["decide"]], h[idx], tq[0])]
        if prefix is None:
            prefix = self.prefill_from_record(rec)
        if prefix["n"] != Ls:
            raise ValueError(f"prefix is {prefix['n']} tokens, record's state is {Ls}")
        Q = len(rows)
        Lb = max(len(r["ids"]) for r in rows)
        bid = torch.full((Q, Lb), self.pad_id, dtype=torch.long, device=self.device)
        bpos = torch.zeros((Q, Lb), dtype=torch.long, device=self.device)
        att = torch.zeros((Q, Ls + Lb), dtype=torch.long, device=self.device)
        att[:, :Ls] = 1
        for i, r in enumerate(rows):
            n = len(r["ids"])
            bid[i, :n] = torch.tensor(r["ids"], device=self.device)
            bpos[i, :n] = torch.tensor(r["pos"], device=self.device)
            att[i, Ls:Ls + n] = 1
        h = self.lm(input_ids=bid, position_ids=bpos, attention_mask=att,
                    past_key_values=self._replicate(prefix["cache"], Q),
                    use_cache=True).last_hidden_state.to(self.head_dtype)
        out = []
        for i, r in enumerate(rows):
            idx = torch.tensor(r["opts"], device=self.device)
            out.append(self.head(h[i, r["decide"]], h[i, idx], tq[i]))
        return out

    @torch.no_grad()
    def row_logits(self, rows):
        """Raw logits for rows that each hold a full state + question sequence: ([B, K_max], validity mask)."""
        B = len(rows)
        L = max(len(r["ids"]) for r in rows)
        Kmax = max(len(r["opts"]) for r in rows)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long, device=self.device)
        att = torch.zeros((B, L), dtype=torch.long, device=self.device)
        oidx = torch.zeros((B, Kmax), dtype=torch.long, device=self.device)
        omask = torch.zeros((B, Kmax), dtype=torch.bool, device=self.device)
        didx = torch.zeros(B, dtype=torch.long, device=self.device)
        for i, r in enumerate(rows):
            n = len(r["ids"])
            ids[i, :n] = torch.tensor(r["ids"], device=self.device)
            att[i, :n] = 1
            k = len(r["opts"])
            oidx[i, :k] = torch.tensor(r["opts"], device=self.device)
            omask[i, :k] = True
            didx[i] = r["decide"]
        h = self.lm(input_ids=ids, attention_mask=att).last_hidden_state.to(self.head_dtype)
        ar = torch.arange(B, device=self.device)
        z = (self.head.k(h[ar[:, None], oidx]) @ self.head.q(h[ar, didx]).unsqueeze(-1)).squeeze(-1) * self.head.scale
        return z.masked_fill(~omask, float("-inf")), omask
