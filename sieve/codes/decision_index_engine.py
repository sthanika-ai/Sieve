"""Decision Index engine for Sieve-4B, in process.

    python -m decision_index run --engine sieve.decision_index_engine:SieveEngine \
        --option model=sthanika-ai/Sieve-4B --out runs/Sieve-4B

Each request is answered exactly as `python -m sieve.codes.server` answers it over HTTP (bf16, adapter merged, both
yes/no orders, four decimals): the response is the server's /v1/systemone body. A request the server refuses for its
size (too many options or levels, too many tokens) is Unsupported; any other failure is an error, retried on resume.
"""
import hashlib
import os

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:                                  # the kit is only needed to run the Decision Index
    from .._kit import Engine, Unsupported

from . import load
from .contract import ContractError, SystemOneRequest

CAPACITY_MARKERS = ("options per choice", "a choice needs at least two options", "a score takes 2 to 10 levels",
                    "maximum context length")


class CodesEngine(Engine):
    name = "sieve-4b"
    latency = ("In-process request wall time: request validation, prompt construction and the model pass; "
               "excludes model loading.")

    def __init__(self, model="sthanika-ai/Sieve-4B", device="cuda:0", served_name="sieve-4b", prefix_cache_gb=1.0,
                 max_batch_tokens=65536, decimals=4, **options):
        super().__init__(model=model, device=device, served_name=served_name, prefix_cache_gb=prefix_cache_gb,
                         max_batch_tokens=max_batch_tokens, decimals=decimals, **options)
        self.served_name = served_name
        self.engine = load(model, device=device, name=served_name, prefix_cache_gb=float(prefix_cache_gb),
                           max_batch_tokens=int(max_batch_tokens), decimals=int(decimals))
        m = self.engine.model
        self.provenance = {"model": os.path.basename(str(model).rstrip("/")), "source": model, "base_model": m.cfg.base,
                           "base_revision": m.cfg.revision, "temperatures": m.temperatures,
                           "serving": "adapter merged into the bf16 backbone; the /v1/systemone response body"}
        if os.path.isdir(model):
            for f in ("adapter_model.safetensors", "head.safetensors"):
                p = os.path.join(model, f)
                if os.path.exists(p):
                    h = hashlib.sha256()
                    with open(p, "rb") as fh:
                        for b in iter(lambda: fh.read(1 << 22), b""):
                            h.update(b)
                    self.provenance[f + "_sha256"] = h.hexdigest()

    def synchronize(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, state, questions):
        try:
            body, stats = self.engine.evaluate(SystemOneRequest(state=state, model=self.served_name, questions=questions))
        except ContractError as e:
            if e.status in (400, 413, 422) and any(s in e.message for s in CAPACITY_MARKERS):
                raise Unsupported(e.message)
            raise RuntimeError(e.message)
        return body, stats
