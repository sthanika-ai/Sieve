"""Sieve as an engine for the Decision Index kit (github.com/apolinario/decision-index).

    python -m decision_index run --engine sieve.decision_index_engine:SieveEngine \
        --option model=sthanika-ai/Sieve-9B --out runs/sieve-9b

The model runs as it is served: adapter merged into the bf16 backbone, fp32 pointer head, calibrated temperature
from head.pt. A single question runs as one causal row (state + question). Several questions share one state
prefill and are answered in batches that fit a memory budget. Nothing is truncated and no option is dropped;
requests over the declared limits raise Unsupported. Running out of GPU memory raises an ordinary error, which
the kit retries when a run is resumed.
"""
import hashlib
import os

import torch
import torch.nn.functional as F

from decision_index.engines import Engine, Unsupported

from .api import MAX_OPTIONS, to_record
from .encode import TooLong, rows_of
from .load import load_sieve

RECURRENT_BYTES_PER_COPY = 32 * 1024 * 1024     # recurrent and convolution state of the linear-attention layers


class SieveEngine(Engine):
    name = "sieve"
    latency = ("In-process wall time of one request, CUDA-synchronized: tokenisation, the backbone pass(es) and the "
               "pointer readout; excludes model loading. Eager PyTorch (the CUDA-graph fast path is not used here).")

    def __init__(self, model, device="cuda", dtype="bfloat16", max_state_tokens=65536, max_question_tokens=32768,
                 memory_budget_gb=6.0, **options):
        super().__init__(model=model, device=device, dtype=dtype, max_state_tokens=max_state_tokens,
                         max_question_tokens=max_question_tokens, memory_budget_gb=memory_budget_gb, **options)
        self.max_state, self.max_branch = int(max_state_tokens), int(max_question_tokens)
        self.budget = float(memory_budget_gb) * 1024 ** 3
        self.m = load_sieve(model, device=device, dtype=getattr(torch, dtype))
        self.T = float(self.m.head.temperature)
        cfg = self.m.lm.config
        attn_layers = sum(t != "linear_attention" for t in (getattr(cfg, "layer_types", None) or [])) \
            or cfg.num_hidden_layers
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.kv_bytes_per_token = attn_layers * 2 * cfg.num_key_value_heads * head_dim * 2
        self.attn_heads = cfg.num_attention_heads
        self.provenance = {"model": os.path.basename(model.rstrip("/")), "source": model, "base_model": self.m.base_model,
                           "temperature": self.T, "serving": "adapter merged into the bf16 backbone, fp32 pointer head, eager"}
        if os.path.isdir(model):
            for f in ("adapter_model.safetensors", "head.pt"):
                p = os.path.join(model, f)
                if os.path.exists(p):
                    h = hashlib.sha256()
                    with open(p, "rb") as fh:
                        for b in iter(lambda: fh.read(1 << 22), b""):
                            h.update(b)
                    self.provenance[f + "_sha256"] = h.hexdigest()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def runtime(self):
        import peft
        import transformers
        d = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
        if torch.cuda.is_available():
            d["gpu"] = torch.cuda.get_device_name(0)
        return d

    @torch.no_grad()
    def __call__(self, state, questions):
        for k, q in (questions or {}).items():
            if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), dict) \
                    and len(q["criteria"]) > MAX_OPTIONS:
                raise Unsupported(f"question {k!r} has {len(q['criteria'])} options (declared limit {MAX_OPTIONS})")
        text, qs, meta = to_record(state, questions)
        try:
            rec = self.m.encode(text, qs, max_state=self.max_state, max_branch=self.max_branch)
        except TooLong as e:
            raise Unsupported(str(e))
        try:
            z = self._logits(text, qs, rec)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            z = None
            if len(qs) > 1:
                try:
                    z = self._logits(text, qs, rec, one_at_a_time=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
            if z is None:
                raise RuntimeError(f"GPU out of memory at {len(rec['ids'])} tokens (retry on resume)")
        answers = {}
        for zi, mi in zip(z, meta):
            p = F.softmax(zi.float() / self.T, -1).cpu().tolist()
            if mi["type"] == "noul":
                answers[mi["id"]] = {"type": "noul", "noul": p[1]}
            else:
                keys = mi["keys"]
                answers[mi["id"]] = {"type": "choice", "choice": keys[max(range(len(p)), key=p.__getitem__)],
                                     "probabilities": dict(zip(keys, p))}
        return {"model": self.provenance["model"].lower(), "answers": answers}, None

    def _logits(self, text, qs, rec, one_at_a_time=False):
        m = self.m
        state_ids, _, rows = rows_of(rec)
        Ls = rec["n_state"]
        if len(rows) == 1:
            r = rows[0]
            row = {"ids": state_ids + r["ids"], "opts": [Ls + o for o in r["opts"]], "decide": Ls + r["decide"]}
            z, _ = m.row_logits([row])
            return [z[0]]
        pre = m.prefill_from_record(rec)
        out, i = [], 0
        while i < len(qs):
            n = 1 if one_at_a_time else self._batch(Ls, [len(r["ids"]) for r in rows[i:]])
            sub = m.encode(text, qs[i:i + n], max_state=self.max_state, max_branch=self.max_branch)
            out += m.logits(sub, prefix=pre, temperature=1.0)
            i += n
        return out

    def _batch(self, Ls, lengths):
        """Largest n such that n replicated caches and n rows of masked attention fit the memory budget."""
        n, Lb = 0, 0
        for L in lengths:
            Lb = max(Lb, L)
            cost = (n + 1) * (Ls * self.kv_bytes_per_token + RECURRENT_BYTES_PER_COPY + self.attn_heads * Lb * (Ls + Lb) * 2)
            if n and cost > self.budget:
                break
            n += 1
        return max(1, n)
