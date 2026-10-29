"""Sieve-31B / Sieve-31B-v2 model computation: Gemma-4-31B-it with the release's weights, read at the answer slots.

The model reads each prompt once. At every answer slot it takes the final hidden state h and scores option j with the
output-layer row of that option's label token: z_j = W[label_j] . h. Label slots past the question's option count are
masked.

- Sieve-31B is a LoRA adapter, applied unmerged (PEFT), exactly as in training and in the submitted run.
- Sieve-31B-v2 is a full fine-tune stored as an adapter (every decoder layer a PEFT modules_to_save module, next to a
  LoRA that is exactly zero, as Sieve-27B-v2). Its layers are copied into the backbone in place (sieve.load).
- A release whose training_config.json says "attention": "bidir_global" (Sieve-31B-v2) was trained with bidirectional
  attention in the 10 global (full-attention) layers: they attend to every prompt token, before and after it, while the
  50 sliding-window layers stay causal.
  It is served the same way. One unpadded row passes no mask to the global layers, so they use the fused non-causal
  SDPA kernel (no extra cost); a padded batch passes a key-padding mask.

The class exposes what the decider-ai orchestrator expects of its model object (`tok`, `lm`, `letters`,
`slot_logits(input_ids, attention_mask, slot_idx, slot_batch, nopts)`), so prompt building and batching stay in that
library, unmodified.
"""
import json
import os

import torch
import torch.nn as nn
from decider.prompt import MAX_OPTIONS, letter_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..load import full_layers_of, load_full_layers

RELEASE_FILES = ["adapter_config.json", "adapter_model.safetensors", "training_config.json"]


def release_dir(adapter):
    """A local folder with the release's files (the folder itself, or a Hub download of just these files)."""
    if os.path.isdir(adapter):
        return adapter
    from huggingface_hub import snapshot_download
    return snapshot_download(adapter, allow_patterns=RELEASE_FILES)


def attention_of(path):
    """The attention the release was trained with: "causal" (default) or "bidir_global" (training_config.json)."""
    try:
        with open(os.path.join(path, "training_config.json")) as f:
            mode = json.load(f).get("attention") or "causal"
    except (OSError, ValueError):
        mode = "causal"
    if mode not in ("causal", "bidir_global"):
        raise ValueError(f"{path}: unknown attention {mode!r}")
    return mode


def set_global_noncausal(lm):
    """Switch is_causal off for the text model's full-attention (global) layers; returns how many."""
    n = 0
    for name, mod in lm.named_modules():
        if hasattr(mod, "is_causal") and getattr(mod, "layer_type", None) == "full_attention" \
                and "vision" not in name and "audio" not in name:
            mod.is_causal = False
            n += 1
    want = sum(1 for t in lm.config.get_text_config().layer_types if t == "full_attention")
    assert n == want, f"{n} global attention modules switched, expected {want}"
    return n


def bidir_global_masks(att, window):
    """attention_mask per layer type for "bidir_global": global layers see every non-pad key (no mask at all when
    nothing is padded), sliding layers are causal within the window."""
    B, L = att.shape
    i = torch.arange(L, device=att.device)
    keys = att.bool()[:, None, :]
    sliding = (((i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window))[None] & keys)[:, None]
    full = None if bool(att.all()) else keys[:, None].expand(B, 1, L, L)
    return {"full_attention": full, "sliding_attention": sliding}


class SlotReadout(nn.Module):
    max_memory = None          # {gpu index: "NGiB"} -> spread the base model over GPUs with accelerate; None -> one device
    adapter = None             # the release: a LoRA adapter or a full fine-tune stored as one (a folder or Hub repo id)
    revision = None            # base model revision

    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=False):
        super().__init__()
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)     # keep SDPA off the cuDNN backend for masked attention
        kwargs = {"dtype": dtype, "revision": self.revision}
        if self.max_memory:
            kwargs.update(device_map="auto", max_memory=self.max_memory)
        self.lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.tok = AutoTokenizer.from_pretrained(name, revision=self.revision)
        self.attention, self.full_release = "causal", False
        if self.adapter:
            path = release_dir(self.adapter)
            self.attention = attention_of(path)
            self.full_release = full_layers_of(path)
            if self.full_release:
                load_full_layers(self.lm, path)                    # Sieve-31B-v2: decoder layers copied in place
            else:
                from peft import PeftModel
                PeftModel.from_pretrained(self.lm, path)           # Sieve-31B: injects the LoRA layers in place
                n = sum(1 for k, _ in self.lm.named_parameters() if "lora_" in k)
                assert n > 0, f"no LoRA weights loaded from {self.adapter}"
            self.lm.eval()
        self._window = self.lm.config.get_text_config().sliding_window
        if self.attention == "bidir_global":
            set_global_noncausal(self.lm)
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)
        self._split = bool(self.max_memory)
        if self._split:
            self.letters = self.letters.to(self.lm.lm_head.weight.device)

    def to(self, *args, **kwargs):
        return self if self._split else super().to(*args, **kwargs)

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        entry = self.lm.get_input_embeddings().weight.device
        am = attention_mask.to(entry)
        if self.attention == "bidir_global":
            am = bidir_global_masks(am, self._window)
        # one pass per prompt: no attention cache (by default the sliding layers would keep the whole prompt's keys and
        # values, ~0.9 GB per 1,000 tokens on Gemma-4-31B); the hidden states are the same
        final = self.lm.model(input_ids=input_ids.to(entry), attention_mask=am, use_cache=False).last_hidden_state
        at_slots = final[slot_batch.to(final.device), slot_idx.to(final.device)]
        rows = self.lm.lm_head.weight[self.letters]
        scores = (at_slots.to(rows.device) @ rows.T).float()
        in_range = torch.arange(MAX_OPTIONS, device=scores.device)[None, :] < nopts.to(scores.device)[:, None]
        return scores.masked_fill(~in_range, float("-inf"))


def make_readout(max_memory=None, adapter=None, revision=None):
    """A SlotReadout class with this configuration baked in (the orchestrator instantiates it itself)."""
    return type("ConfiguredSlotReadout", (SlotReadout,), {"max_memory": max_memory, "adapter": adapter, "revision": revision})
