"""Load any Sieve model and answer typed decision requests.

Sieve models come in five designs, told apart by their files:

    pointer  head.pt                    Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3
                                                                            (sieve.model)
    fork     config.json (sieve-hybrid-fork) + head.safetensors
                                        Sieve-2B-v2, Sieve-2B, Sieve-0.8B   (sieve.fork)
    codes    sieve_config.json          Sieve-4B                            (sieve.codes)
    slots    adapter_config.json on Gemma-4-31B-it, no head
                                        Sieve-31B-v2, Sieve-31B             (sieve.slots)
    letters  a full model (config.json + safetensors) and head.pt with readout "letters", no adapter
                                        Sieve-31B-v3                        (sieve.letters)

load_sieve and decide work with all of them.
"""
import json
import os

import torch
import torch.nn.functional as F

from .api import to_answers, to_record
from .model import SieveModel

BASE_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


DEFAULT_LIMITS = (8192, 8192)


def _resolve(path_or_repo):
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, allow_patterns=["adapter_config.json", "adapter_model.safetensors", "head.pt",
                                                           "training_config.json"])


def limits_of(path):
    """The token limits a Sieve model was trained with, from its training_config.json: (state, question with its
    options and delimiters). Models that do not record them get DEFAULT_LIMITS."""
    try:
        with open(os.path.join(path, "training_config.json")) as f:
            recipe = json.load(f).get("recipe") or {}
    except (OSError, ValueError):
        return DEFAULT_LIMITS
    return (int(recipe.get("max_state_tokens") or DEFAULT_LIMITS[0]),
            int(recipe.get("max_question_and_options_tokens") or DEFAULT_LIMITS[1]))


def base_of(path):
    """The backbone a Sieve model was trained on (from its adapter_config.json) and its pinned revision."""
    with open(os.path.join(path, "adapter_config.json")) as f:
        base = json.load(f)["base_model_name_or_path"]
    return base, BASE_REVISIONS.get(base)


def full_layers_of(path):
    """True for a full fine-tune stored as an adapter (Sieve-27B-v2, Sieve-2B-v3): trained modules as PEFT modules_to_save and a LoRA
    that is exactly zero. Its weights can be copied into the backbone directly; ordinary LoRA adapters give False."""
    with open(os.path.join(path, "adapter_config.json")) as f:
        if not json.load(f).get("modules_to_save"):
            return False
    from safetensors import safe_open
    with safe_open(os.path.join(path, "adapter_model.safetensors"), "pt", device="cpu") as f:
        lora = {}
        for k in f.keys():
            if ".lora_" in k:
                lora.setdefault(k.split(".lora_")[0], []).append(f.get_tensor(k))
    return all(any(not t.any() for t in ts) for ts in lora.values())


@torch.no_grad()
def load_full_layers(lm, path):
    """Copy a full fine-tune's trained modules into the backbone in place, tensor by tensor. The result is the model
    PEFT builds (merged or not, as its LoRA is zero), without PEFT's copy of every trained module: for a 27B that copy
    would need about 2.5x the layers' size on the GPU on top of the backbone."""
    from safetensors import safe_open
    params = lm.state_dict(keep_vars=True)          # parameters and persistent buffers, as PEFT saves them
    with open(os.path.join(path, "adapter_config.json")) as f:
        modules = json.load(f)["modules_to_save"]
    want = {n for n in params if any(n.startswith(m + ".") for m in modules)}
    done = set()
    with safe_open(os.path.join(path, "adapter_model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            if ".lora_" in k:
                continue
            n = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
            if n not in want:
                raise ValueError(f"{path}: {k} is not a tensor of a module to save")
            params[n].copy_(f.get_tensor(k))
            done.add(n)
    if done != want:
        raise ValueError(f"{path}: {len(want - done)} tensors of the modules to save are missing, e.g. {sorted(want - done)[:3]}")


def load_pointer(path_or_repo, base=None, base_revision=None, device="cuda", dtype=torch.bfloat16,
                 merge=True, temperature=None):
    """Load a pointer-head model's adapter and head from a local folder or a Hugging Face repo onto its pinned backbone.

    base and base_revision default to the backbone named in the model's adapter_config.json.

    The model's trained token limits (limits_of) become its encode defaults, used by decide and the server.

    merge=True folds the adapter into the backbone for speed; merge=False keeps the exact unmerged forward. A full
    fine-tune (full_layers_of) is copied into the backbone in place either way: both forwards are that model.
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
    full = full_layers_of(path)
    if full:
        load_full_layers(m.lm, path)
    else:
        m.lm = PeftModel.from_pretrained(m.lm, path, is_trainable=False)
    m.load_head(path)
    m.max_state, m.max_branch = limits_of(path)
    if temperature is not None:
        m.head.temperature = float(temperature)
        m.temperatures = None
    if merge and not full:
        m.lm = m.lm.merge_and_unload()
    m.lm.eval()
    m.head.eval()
    return m


@torch.no_grad()
def decide_pointer(m, state, questions, temperature=None, prefix=None):
    """Answer every question about one state with a pointer-head model.

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


FAMILY_FILES = ["head.pt", "config.json", "sieve_config.json", "adapter_config.json"]
SLOTS_BASES = {"google/gemma-4-31B-it"}          # answer-slot models: an adapter on one of these, no head file


def is_letters(path):
    """A letter-readout release: head.pt says readout "letters" and there is no adapter (every pointer-head model is an
    adapter with head.pt, so its head.pt is never opened here)."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return False
    try:
        meta = torch.load(os.path.join(path, "head.pt"), map_location="cpu", weights_only=True)
    except Exception:
        return False
    return isinstance(meta, dict) and meta.get("readout") == "letters"


def family_of(path_or_repo):
    """The design of a Sieve model (pointer, fork, codes, slots or letters), from the files of its folder or Hub repo."""
    if os.path.isdir(path_or_repo):
        path = path_or_repo
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(path_or_repo, allow_patterns=FAMILY_FILES)
    has = lambda f: os.path.exists(os.path.join(path, f))
    if has("head.pt"):
        return "letters" if is_letters(path) else "pointer"
    if has("sieve_config.json"):
        return "codes"
    if has("config.json"):
        with open(os.path.join(path, "config.json")) as f:
            if json.load(f).get("architecture") == "sieve-hybrid-fork":
                return "fork"
    if has("adapter_config.json"):
        with open(os.path.join(path, "adapter_config.json")) as f:
            if json.load(f).get("base_model_name_or_path") in SLOTS_BASES:
                return "slots"
        raise ValueError(f"{path_or_repo}: an adapter without its head (head.pt, head.safetensors) or Sieve config; "
                         "is the download complete?")
    raise ValueError(f"{path_or_repo}: not a Sieve model (no head.pt, sieve_config.json, Sieve config.json "
                     "or adapter_config.json)")


def load_sieve(path_or_repo, base=None, base_revision=None, device="cuda", dtype=torch.bfloat16, merge=True,
               temperature=None, **options):
    """Load any Sieve model from a local folder or a Hugging Face repo.

    The arguments of a pointer-head model (Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3) are those of load_pointer. temperature
    overrides any model's calibrated temperatures (1.0 gives raw probabilities). Other options go to the model's own
    loader: prefix_cache_gb for Sieve-4B; revision for the option-fork models; split, max_tokens and fwd_tokens for
    Sieve-31B, whose base and base_revision are its backbone; max_tokens and budget_tokens for Sieve-31B-v3.
    """
    family = family_of(path_or_repo)
    if family == "letters":
        from .letters.engine import LettersEngine
        e = LettersEngine(model=path_or_repo, device=device, dtype=str(dtype).replace("torch.", ""), **options)
        if temperature is not None:
            e.m.temperature, e.m.temperatures = float(temperature), None
        return e
    if family == "pointer":
        return load_pointer(path_or_repo, base=base, base_revision=base_revision, device=device, dtype=dtype,
                            merge=merge, temperature=temperature)
    if family == "fork":
        from .fork import Sieve
        m = Sieve.from_pretrained(path_or_repo, device=device, dtype=dtype, revision=options.pop("revision", None))
        if temperature is not None:
            m.T = float(temperature)
        return m
    if family == "codes":
        from .codes import load
        e = load(path_or_repo, device="cuda:0" if device == "cuda" else device, merge=merge, **options)
        if temperature is not None:
            e.model.temperatures = {t: float(temperature) for t in e.model.temperatures}
        return e
    from .slots.engine import SlotsEngine
    if temperature is not None:
        options["temperature"] = temperature
    if base is not None:
        options["base"] = base
    if base_revision is not None:
        options["revision"] = base_revision
    return SlotsEngine(model=path_or_repo, device=device, **options)


def decide(m, state, questions, temperature=None, prefix=None):
    """Answer every question about one state, with any model from load_sieve.

    state: str | dict | list. questions: {id: {"type": "choice" | "noul" | "score", "instructions": ..., "criteria": ...}}.
    Returns {id: answer}; every answer has its "type" and either "choice", "confidence" and "probabilities" (choice),
    "noul" = P(yes) (noul) or "score", "confidence", "legend" and "probabilities" (score). Sieve-0.8B, Sieve-2B and
    Sieve-2B-v2 answer choice and noul questions; temperature here works with pointer models only.
    """
    if isinstance(m, SieveModel):
        return decide_pointer(m, state, questions, temperature=temperature, prefix=prefix)
    if temperature is not None:
        raise ValueError("this model takes its temperature at load time: load_sieve(..., temperature=...)")
    from .fork import Sieve
    if isinstance(m, Sieve):
        from .fork import decide as decide_fork
        return decide_fork(m, state, questions)
    from .codes.engine import Engine as CodesServing
    if isinstance(m, CodesServing):
        from .codes import decide as decide_codes
        return decide_codes(m, state, questions)["answers"]
    response, _ = m(state, questions)                  # Sieve-31B: its Decision Index engine answers the request
    return response["answers"]
