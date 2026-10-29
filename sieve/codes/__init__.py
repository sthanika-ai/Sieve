"""Sieve-4B: a typed-decision model. A state and typed questions go in; a calibrated probability for every option
of every question comes out. It never generates text.

    from sieve.codes import load, decide
    engine = load("sthanika-ai/Sieve-4B")    # a Hub id or a local copy; fetches Qwen/Qwen3.5-4B on first use
    decide(engine, state, questions)          # the /v1/systemone response body: {"answers": {...}, "usage": {...}}
"""
from pathlib import Path

FILES = ["sieve_config.json", "adapter_config.json", "adapter_model.safetensors", "head.safetensors"]


def resolve(repo_or_path: str) -> Path:
    """A local folder as it is, or the model files of a Hub repo (downloaded once, then cached)."""
    path = Path(repo_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(repo_or_path), allow_patterns=FILES))


def load(repo_or_path: str = "sthanika-ai/Sieve-4B", device: str = "cuda:0", name: str = "sieve-4b",
         prefix_cache_gb: float = 1.0, merge: bool = True, **engine_kw):
    from .engine import Engine
    from .model import DecisionModel, PrefixCache
    model = DecisionModel.load(resolve(repo_or_path), device=device, merge=merge).eval()
    model._prefix_cache = PrefixCache(max_bytes=int(prefix_cache_gb * 2**30))
    return Engine(model, model_name=name, **engine_kw)


def as_content(state, questions):
    """(state, questions) with every bare number or bool (the state, an instructions, a description, a level) as its
    text, str(x) as sieve.api.render writes it: the request schema takes only text, objects and lists."""
    from ..api import render
    c = lambda x: render(x) if isinstance(x, (bool, int, float)) else x
    out = {}
    for k, q in questions.items():
        q = {**q, **({"instructions": c(q["instructions"])} if "instructions" in q else {})}   # absent stays absent
        crit = q.get("criteria")
        if isinstance(crit, dict):
            q["criteria"] = {ck: c(cv) for ck, cv in crit.items()}
        elif isinstance(crit, list):
            q["criteria"] = [c(x) for x in crit]
        out[k] = q
    return c(state), out


def decide(engine, state, questions: dict, model: str = "sieve-4b") -> dict:
    from .contract import SystemOneRequest
    state, questions = as_content(state, questions)
    body, _ = engine.evaluate(SystemOneRequest(state=state, model=model, questions=questions))
    return body
