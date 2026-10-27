"""A letter-readout release (Sieve-31B-v3) as an engine for the Decision Index kit (github.com/apolinario/decision-index).
sieve.decision_index_engine:SieveEngine hands a letter-readout model to this engine:

    python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=<release> --out runs/x

The model runs as served by sieve.letters.model: bf16 weights, attention rule A in the global layers, the release's
serving policy (head.pt "serving") and calibration temperatures (per question category). Nothing is truncated and no
option is dropped; a request over max_tokens (default: none) raises Unsupported. Running out of GPU memory raises an
ordinary error, which the kit retries when a run is resumed."""
import hashlib
import os

import torch

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:                                  # the kit is only needed to run the Decision Index
    from .._kit import Engine, Unsupported

from ..api import MAX_OPTIONS, to_answers, to_record
from ..calibration import question_temperatures
from .model import LetterModel
from .model import Unsupported as TooLong


def resolve(model):
    """A local release folder as it is, or a Hugging Face repo id downloaded (the whole release)."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


class LettersEngine(Engine):
    name = "sieve-letters"
    latency = ("In-process wall time of one request, CUDA-synchronized: tokenisation, the backbone pass(es) under attention "
               "rule A and the letter readout; excludes model loading. Eager PyTorch.")

    def __init__(self, model, device="cuda", dtype="bfloat16", max_tokens=None, budget_tokens=16384, **options):
        super().__init__(model=model, device=device, dtype=dtype, max_tokens=max_tokens, budget_tokens=budget_tokens, **options)
        path = resolve(model)
        self.m = LetterModel.load(path, device=device, dtype=getattr(torch, dtype),
                                  max_tokens=int(max_tokens) if max_tokens else None)
        self.budget = int(budget_tokens)
        self.provenance = {"model": os.path.basename(model.rstrip("/")), "source": model, "temperature": self.m.temperature,
                           "global_attention": "bidirectional (rule A)" if self.m.bidir else "causal",
                           "serving": self.m.policy}
        if self.m.temperatures:
            self.provenance["temperatures"] = self.m.temperatures
        p = os.path.join(path, "head.pt")
        if os.path.exists(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""):
                    h.update(b)
            self.provenance["head.pt_sha256"] = h.hexdigest()

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def runtime(self):
        import transformers
        d = {"torch": torch.__version__, "transformers": transformers.__version__}
        if torch.cuda.is_available():
            d["gpu"] = torch.cuda.get_device_name(0)
        return d

    @torch.no_grad()
    def __call__(self, state, questions):
        for k, q in (questions or {}).items():
            if isinstance(q, dict) and q.get("type") == "choice" and isinstance(q.get("criteria"), dict) \
                    and len(q["criteria"]) > MAX_OPTIONS:
                raise Unsupported(f"question {k!r} has {len(q['criteria'])} options (declared limit {MAX_OPTIONS})")
        _, _, meta = to_record(state, questions)
        try:
            z = self.m.logits_many([(state, questions)], budget=self.budget)[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory (retry on resume)")
        if isinstance(z, TooLong):
            raise Unsupported(str(z))
        temps = question_temperatures(self.m.temperatures, meta, self.m.temperature, state) or [self.m.temperature] * len(meta)
        answers = {}
        for zi, mi, t in zip(z, meta, temps):
            p = torch.softmax(zi.float() / t, -1).tolist()
            if mi["type"] == "noul":
                answers[mi["id"]] = {"type": "noul", "noul": p[1]}
            elif mi["type"] == "score":                     # levels in order (no score question in the Decision Index)
                answers.update(to_answers([p], [mi]))
            else:
                keys = mi["keys"]
                answers[mi["id"]] = {"type": "choice", "choice": keys[max(range(len(p)), key=p.__getitem__)],
                                     "probabilities": dict(zip(keys, p))}
        return {"model": self.provenance["model"], "answers": answers}, None
