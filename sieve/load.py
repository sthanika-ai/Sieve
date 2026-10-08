"""Load a Sieve model and answer typed decision requests."""
import json
import os

import torch
import torch.nn.functional as F

from .api import to_answers, to_record
from .model import SieveModel

BASE_REVISIONS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


def _resolve(path_or_repo):
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, allow_patterns=["adapter_config.json", "adapter_model.safetensors", "head.pt"])


def base_of(path):
    """The backbone a Sieve model was trained on (from its adapter_config.json) and its pinned revision."""
    with open(os.path.join(path, "adapter_config.json")) as f:
        base = json.load(f)["base_model_name_or_path"]
    return base, BASE_REVISIONS.get(base)


def load_sieve(path_or_repo, base=None, base_revision=None, device="cuda", dtype=torch.bfloat16,
               merge=True, temperature=None):
    """Load the adapter and head from a local folder or a Hugging Face repo onto the model's pinned backbone.

    base and base_revision default to the backbone named in the model's adapter_config.json.

    merge=True folds the adapter into the backbone for speed; merge=False keeps the exact unmerged forward.
    temperature overrides the calibrated temperatures stored in head.pt, the global one and any per-category ones,
    for every question (1.0 gives raw logits).
    """
    from peft import PeftModel
    path = _resolve(path_or_repo)
    if base is None:
        base, pinned = base_of(path)
        base_revision = base_revision or pinned
    name = base
    if base_revision and not os.path.isdir(base):
        from huggingface_hub import snapshot_download
        name = snapshot_download(base, revision=base_revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    m = SieveModel(name, device=device, dtype=dtype)
    m.base_model = base
    m.lm = PeftModel.from_pretrained(m.lm, path, is_trainable=False)
    m.load_head(path)
    if temperature is not None:
        m.head.temperature = float(temperature)
        m.temperatures = None
    if merge:
        m.lm = m.lm.merge_and_unload()
    m.lm.eval()
    m.head.eval()
    return m


@torch.no_grad()
def decide(m, state, questions, temperature=None, prefix=None):
    """Answer every question about one state.

    state: str | dict | list. questions: {id: {"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...}}.
    """
    if temperature is not None and not temperature > 0:
        raise ValueError("temperature must be > 0 (1.0 = raw logits; None = the calibrated default)")
    text, qs, meta = to_record(state, questions)
    rec = m.encode(text, qs)
    if temperature is None:                         # the calibrated default: per category when head.pt has a table
        temperature = m.question_temperatures(meta, state)
    probs = [F.softmax(z, -1).float().cpu() for z in m.logits(rec, prefix=prefix, temperature=temperature)]
    return to_answers(probs, meta)
