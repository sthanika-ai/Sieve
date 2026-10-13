"""DecisionModel: pretrained causal decoder + listwise option readout, run through the state-once graph.

Readout: at each branch's last token, z = W h with W in R^{255 x d} initialised from the base's lm_head rows of the
255 single-token option codes, masked to the question's K options, softmax(z / T_type). The last position has read the
whole option list, so the readout is listwise.
Ablations (off by default, adopted only if they win on held-out data):
  pointer: z_j += alpha * (Wq h_last) . (Wk h_end(opt_j)) / 16, alpha initialised to 0 (bilinear form)
  noul_head: a scalar yes-vs-no logit initialised to lm_head[Yes-code] - lm_head[No-code]; not used by default.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from .prompt import MAX_OPTIONS, Branch, Encoded, PromptBuilder
from .shared_prefix import encode_prefix, forward_shared, run_branches, text_model

TYPES = ("choice", "score", "noul")


@dataclass
class ModelConfig:
    base: str
    revision: str | None = None
    pointer: bool = False
    pointer_dim: int = 256
    temperatures: dict | None = None
    preamble: str | None = None


def _dtype_for(device: str):
    return torch.bfloat16 if str(device).startswith("cuda") else torch.float32


class DecisionModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, device: str = "cpu", dtype=None, quantization: str | None = None,
                 device_map=None, attn_implementation: str = "sdpa", load_backbone: bool = True):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base, revision=cfg.revision)
        self.prompts = PromptBuilder(self.tokenizer, **({"preamble": cfg.preamble} if cfg.preamble else {}))
        self.temperatures = dict(cfg.temperatures or {t: 1.0 for t in TYPES})
        dtype = dtype or _dtype_for(device)
        kw = dict(dtype=dtype, attn_implementation=attn_implementation, revision=cfg.revision)
        if quantization == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif quantization == "int8":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if device_map is not None:
            kw["device_map"] = device_map
        lm = AutoModelForCausalLM.from_pretrained(cfg.base, **kw)
        head = lm.get_output_embeddings().weight
        d = head.shape[1]
        ids = torch.tensor(self.prompts.code_ids, device=head.device)
        self.readout = nn.Linear(d, MAX_OPTIONS, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.readout.weight.copy_(head.index_select(0, ids).float())
        self.backbone = lm.model
        lm.lm_head = None
        del lm
        self.pointer = None
        if cfg.pointer:
            self.pointer = nn.ModuleDict({"q": nn.Linear(d, cfg.pointer_dim, bias=False),
                                          "k": nn.Linear(d, cfg.pointer_dim, bias=False)})
            self.pointer_alpha = nn.Parameter(torch.zeros(()))
        self.hidden = d
        if device_map is None:
            self.backbone.to(device)
        self.readout.to(self.head_device)
        if self.pointer is not None:
            self.pointer.to(self.head_device); self.pointer_alpha.data = self.pointer_alpha.data.to(self.head_device)
        self._prefix_cache = PrefixCache(max_bytes=4 << 30)

    # ----------------------------------------------------------------------------------------------- properties
    @property
    def input_device(self):
        return text_model(self.backbone).embed_tokens.weight.device

    @property
    def head_device(self):
        return text_model(self.backbone).norm.weight.device

    @property
    def pad_id(self) -> int:
        t = self.tokenizer
        return t.pad_token_id if t.pad_token_id is not None else t.eos_token_id

    # -------------------------------------------------------------------------------------------------- logits
    def _logits(self, h_last: torch.Tensor, h_opts: list[torch.Tensor] | None, n_opts: Sequence[int],
                weight: torch.Tensor | None = None) -> list[torch.Tensor]:
        z = self.readout(h_last.float()) if weight is None else torch.nn.functional.linear(h_last.float(), weight)
        out = []
        for i, K in enumerate(n_opts):
            zi = z[i, :K]
            if self.pointer is not None and h_opts is not None and weight is None:
                q = self.pointer["q"](h_last[i].float())
                k = self.pointer["k"](h_opts[i].float())
                zi = zi + self.pointer_alpha * (k @ q) / math.sqrt(k.shape[-1])
            out.append(zi)
        return out

    # -------------------------------------------------------------------------------------------- training path
    def forward_records(self, records: Sequence[Encoded], readout_weight: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Records -> per-branch logits (listing order), in record order then branch order. `readout_weight` overrides
        the readout (e.g. the frozen initial rows, for the replay-KL anchor)."""
        dev, pad = self.input_device, self.pad_id
        R = len(records)
        Ls = max(len(r.prefix_ids) for r in records)
        branches = [b for r in records for b in r.branches]
        owner = [ri for ri, r in enumerate(records) for _ in r.branches]
        Lb = max(len(b.ids) for b in branches)
        s_ids = torch.full((R, Ls), pad, dtype=torch.long); s_pos = torch.zeros((R, Ls), dtype=torch.long)
        s_real = torch.zeros((R, Ls), dtype=torch.bool)
        for i, r in enumerate(records):                                      # left-padded states
            n = len(r.prefix_ids)
            s_ids[i, Ls - n:] = torch.tensor(r.prefix_ids); s_pos[i, Ls - n:] = torch.arange(n); s_real[i, Ls - n:] = True
        B = len(branches)
        b_ids = torch.full((B, Lb), pad, dtype=torch.long); b_pos = torch.zeros((B, Lb), dtype=torch.long)
        b_real = torch.zeros((B, Lb), dtype=torch.bool)
        for i, (b, ri) in enumerate(zip(branches, owner)):                   # right-padded branches
            n, start = len(b.ids), len(records[ri].prefix_ids)
            b_ids[i, :n] = torch.tensor(b.ids); b_pos[i, :n] = start + torch.arange(n); b_real[i, :n] = True
        t = lambda x: x.to(dev)
        h = forward_shared(self.backbone, t(s_ids), t(s_pos), t(s_real), t(b_ids), t(b_pos), t(b_real),
                           torch.tensor(owner, device=dev))
        h = h.to(self.head_device)
        last = torch.tensor([b.readout for b in branches], device=h.device)
        h_last = h[torch.arange(B, device=h.device), last]
        h_opts = None
        if self.pointer is not None:
            h_opts = [h[i, torch.tensor(b.opt_ends, device=h.device)] for i, b in enumerate(branches)]
        return self._logits(h_last, h_opts, [b.n_opts for b in branches], readout_weight)

    def forward(self, records: Sequence[Encoded]) -> list[torch.Tensor]:      # DDP calls module(...)
        return self.forward_records(records)

    # --------------------------------------------------------------------------------------------- serving path
    @torch.inference_mode()
    def branch_logits(self, enc: Encoded, max_batch_tokens: int = 1 << 17) -> list[torch.Tensor]:
        """One request: state once (LRU-cached by token hash), branches in chunks bounded by max_batch_tokens
        (a branch's attention covers prefix + itself)."""
        dev = self.input_device
        key = hash(tuple(enc.prefix_ids))
        prefixes = self._prefix_cache.get(key)
        if prefixes is None:
            prefixes = encode_prefix(self.backbone, torch.tensor([enc.prefix_ids], device=dev))
            self._prefix_cache.put(key, prefixes)
        Ls = len(enc.prefix_ids)
        order = sorted(range(len(enc.branches)), key=lambda i: len(enc.branches[i].ids))
        out: list[torch.Tensor | None] = [None] * len(enc.branches)
        i = 0
        while i < len(order):
            Lb = len(enc.branches[order[i]].ids)
            j = i
            while j < len(order) and (j - i + 1) * (Ls + len(enc.branches[order[j]].ids)) <= max_batch_tokens:
                j += 1
            j = max(j, i + 1)
            chunk = [enc.branches[k] for k in order[i:j]]
            Lb = max(len(b.ids) for b in chunk)
            G = 1 + (max(b.n_opts for b in chunk) if self.pointer is not None else 0)
            b_ids = torch.full((len(chunk), Lb), self.pad_id, dtype=torch.long)
            b_real = torch.zeros((len(chunk), Lb), dtype=torch.bool)
            gidx = torch.zeros((len(chunk), G), dtype=torch.long)
            for r, b in enumerate(chunk):
                b_ids[r, :len(b.ids)] = torch.tensor(b.ids); b_real[r, :len(b.ids)] = True
                gidx[r, 0] = b.readout
                if self.pointer is not None:
                    gidx[r, 1:1 + b.n_opts] = torch.tensor(b.opt_ends)
            h = run_branches(self.backbone, prefixes, Ls, b_ids.to(dev), b_real.to(dev), gidx.to(dev)).to(self.head_device)
            h_opts = [h[r, 1:1 + b.n_opts] for r, b in enumerate(chunk)] if self.pointer is not None else None
            for k, z in zip(order[i:j], self._logits(h[:, 0], h_opts, [b.n_opts for b in chunk])):
                out[k] = z
            i = j
        return out

    def probabilities(self, logits: torch.Tensor, qtype: str) -> list[float]:
        return torch.softmax(logits.float() / self.temperatures.get(qtype, 1.0), -1).tolist()

    # ---------------------------------------------------------------------------------------------- save / load
    def trainable_head_state(self) -> dict:
        sd = {"readout.weight": self.readout.weight.detach().cpu()}
        if self.pointer is not None:
            sd.update({f"pointer.{k}": v.detach().cpu() for k, v in self.pointer.state_dict().items()})
            sd["pointer_alpha"] = self.pointer_alpha.detach().cpu()
        return sd

    def save(self, out: str | Path, extra: dict | None = None):
        from safetensors.torch import save_file
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained") and hasattr(self.backbone, "peft_config"):
            self.backbone.save_pretrained(str(out / "adapter"))
        save_file({k: v.contiguous() for k, v in self.trainable_head_state().items()}, str(out / "head.safetensors"))
        meta = {"base": self.cfg.base, "revision": self.cfg.revision, "pointer": self.cfg.pointer,
                "pointer_dim": self.cfg.pointer_dim, "temperatures": self.temperatures, "codes": self.prompts.codes,
                "code_ids": self.prompts.code_ids, "preamble": self.prompts.preamble, **(extra or {})}
        (out / "sieve_config.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, ckpt: str | Path, **kw) -> "DecisionModel":
        from safetensors.torch import load_file
        ckpt = Path(ckpt)
        meta = json.loads((ckpt / "sieve_config.json").read_text())
        cfg = ModelConfig(base=meta["base"], revision=meta.get("revision"), pointer=meta.get("pointer", False),
                          pointer_dim=meta.get("pointer_dim", 256), temperatures=meta.get("temperatures"),
                          preamble=meta.get("preamble"))
        merge = kw.pop("merge", True)
        m = cls(cfg, **kw)
        if m.prompts.code_ids != meta["code_ids"]:
            raise ValueError("checkpoint option codes differ from this tokenizer")
        adapter = ckpt / "adapter" if (ckpt / "adapter").exists() else (ckpt if (ckpt / "adapter_config.json").exists() else None)
        if adapter is not None:
            from peft import PeftModel
            m.backbone = PeftModel.from_pretrained(m.backbone, str(adapter), is_trainable=not merge)
            if merge and not kw.get("quantization"):
                m.backbone = m.backbone.merge_and_unload()
        sd = load_file(str(ckpt / "head.safetensors"))
        m.readout.weight.data.copy_(sd["readout.weight"].to(m.readout.weight.device))
        if m.pointer is not None:
            m.pointer.load_state_dict({k[len("pointer."):]: v for k, v in sd.items() if k.startswith("pointer.")})
            m.pointer_alpha.data.copy_(sd["pointer_alpha"])
        return m


class PrefixCache:
    """LRU over encoded states, bounded in bytes: a repeated state skips prefill."""

    def __init__(self, max_bytes: int):
        from collections import OrderedDict
        self.max_bytes, self.d, self.bytes = max_bytes, OrderedDict(), 0

    def get(self, key):
        v = self.d.get(key)
        if v is not None:
            self.d.move_to_end(key)
        return v

    def put(self, key, prefixes):
        n = sum(p.nbytes() for p in prefixes)
        if n > self.max_bytes:
            return
        self.d[key] = prefixes; self.bytes += n
        while self.bytes > self.max_bytes:
            _, old = self.d.popitem(last=False)
            self.bytes -= sum(p.nbytes() for p in old)

    def clear(self):
        self.d.clear(); self.bytes = 0
