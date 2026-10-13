"""Sieve-31B model computation: the base model with the LoRA adapter, read at the answer slots.

The model reads each prompt once. At every answer slot it takes the final hidden state h and scores option j with the
output-layer row of that option's label token: z_j = W[label_j] . h. Label slots past the question's option count are
masked. The adapter is applied unmerged (PEFT), exactly as in training and in the submitted run.

The class exposes what the decider-ai orchestrator expects of its model object (`tok`, `lm`, `letters`,
`slot_logits(input_ids, attention_mask, slot_idx, slot_batch, nopts)`), so prompt building and batching stay in that
library, unmodified.
"""
import torch
import torch.nn as nn
from decider.prompt import MAX_OPTIONS, letter_ids
from transformers import AutoModelForCausalLM, AutoTokenizer


class SlotReadout(nn.Module):
    max_memory = None          # {gpu index: "NGiB"} -> spread the base model over GPUs with accelerate; None -> one device
    adapter = None             # the LoRA adapter (a folder or Hub repo id)
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
        if self.adapter:
            from peft import PeftModel
            PeftModel.from_pretrained(self.lm, self.adapter)      # injects the LoRA layers into self.lm in place
            n = sum(1 for k, _ in self.lm.named_parameters() if "lora_" in k)
            assert n > 0, f"no LoRA weights loaded from {self.adapter}"
            self.lm.eval()
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)
        self._split = bool(self.max_memory)
        if self._split:
            self.letters = self.letters.to(self.lm.lm_head.weight.device)

    def to(self, *args, **kwargs):
        return self if self._split else super().to(*args, **kwargs)

    def slot_logits(self, input_ids, attention_mask, slot_idx, slot_batch, nopts):
        entry = self.lm.get_input_embeddings().weight.device
        final = self.lm.model(input_ids=input_ids.to(entry), attention_mask=attention_mask.to(entry)).last_hidden_state
        at_slots = final[slot_batch.to(final.device), slot_idx.to(final.device)]
        rows = self.lm.lm_head.weight[self.letters]
        scores = (at_slots.to(rows.device) @ rows.T).float()
        in_range = torch.arange(MAX_OPTIONS, device=scores.device)[None, :] < nopts.to(scores.device)[:, None]
        return scores.masked_fill(~in_range, float("-inf"))


def make_readout(max_memory=None, adapter=None, revision=None):
    """A SlotReadout class with this configuration baked in (the orchestrator instantiates it itself)."""
    return type("ConfiguredSlotReadout", (SlotReadout,), {"max_memory": max_memory, "adapter": adapter, "revision": revision})
