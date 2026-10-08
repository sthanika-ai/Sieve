"""HTTP server for Sieve.

    python -m sieve.serve --model sthanika-ai/Sieve-9B --graphs --port 8020

POST /v1/decide  {"state": ..., "questions": {...}, "temperature": optional}  ->  typed answers with probabilities
GET  /v1/models  loaded model, temperature, CUDA-graph and prefix-cache state
"""
import argparse
import threading
import time
from collections import OrderedDict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .api import to_answers, to_record

app = FastAPI(title="sieve")


class Request(BaseModel):
    state: object
    questions: dict
    temperature: float | None = None


class Server:
    def __init__(self, model, cache_size=8, min_state_tokens=64):
        self.m = model
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.size, self.min_tokens = cache_size, min_state_tokens
        self.hits = self.misses = 0
        self.graphed = None
        self.model_id = self.base = None

    def prefix_for(self, rec):
        """Cached state prefill for this exact state, keyed by its token ids."""
        Ls = rec["n_state"]
        if not self.size or Ls < self.min_tokens:
            return None, False
        key = tuple(rec["ids"][:Ls])
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key], True
        pre = self.m.prefill_from_record(rec)
        self.cache[key] = pre
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        self.misses += 1
        return pre, False

    def decide(self, req: Request):
        if req.temperature is not None and not req.temperature > 0:
            raise HTTPException(422, "temperature must be > 0 (1.0 = raw logits; omit it for the calibrated default)")
        try:
            state, qs, meta = to_record(req.state, req.questions)
            rec = self.m.encode(state, qs)
        except ValueError as e:
            raise HTTPException(422, str(e))
        t = req.temperature if req.temperature is not None else self.m.question_temperatures(meta, req.state)
        with self.lock:
            if self.m.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pre, hit = self.prefix_for(rec)
            if self.graphed is not None:
                if pre is None:
                    pre = self.m.prefill_from_record(rec)
                logits = self.graphed.logits(rec, prefix=pre, temperature=t)
            else:
                logits = self.m.logits(rec, prefix=pre, temperature=t)
            probs = [torch.softmax(z, -1).cpu() for z in logits]
            if self.m.device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
        return {"answers": to_answers(probs, meta),
                "usage": {"input_tokens": len(rec["ids"]), "state_tokens": rec["n_state"],
                          "questions": len(qs), "output_tokens": 0},
                "latency_ms": round(dt, 1), "prefix_cache_hit": hit}


@app.post("/v1/decide")
def decide(req: Request):
    return app.state.server.decide(req)


@app.get("/v1/models")
def models():
    s = app.state.server
    return {"model": s.model_id, "base": s.base, "device": str(s.m.device),
            "temperature": s.m.head.temperature,
            "temperatures_by_category": (s.m.temperatures or {}).get("table"),
            "cuda_graphs": None if s.graphed is None else {"captured": len(s.graphed.graphs),
                                                           "max_graph_tokens": s.graphed.max_graph_tokens},
            "prefix_cache": {"size": s.size, "hits": s.hits, "misses": s.misses, "cached_states": len(s.cache)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local folder or Hugging Face repo id")
    ap.add_argument("--base", default=None, help="backbone (default: the one in the model's adapter_config.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--no-merge", action="store_true", help="keep the adapter unmerged (exact, but slower)")
    ap.add_argument("--graphs", action="store_true", help="replay short requests as CUDA graphs")
    ap.add_argument("--graph-tokens", type=int, default=0,
                    help="largest questions x branch tokens replayed as a graph (0 = 1024 for hybrid backbones, else 512)")
    a = ap.parse_args()

    from .load import load_sieve
    m = load_sieve(a.model, base=a.base, merge=not a.no_merge)
    srv = Server(m)
    srv.model_id, srv.base = a.model, m.base_model
    if a.graphs:
        from .graphs import CudaGraphRunner
        hybrid = "linear_attention" in set(getattr(m.lm.config, "layer_types", None) or [])
        srv.graphed = CudaGraphRunner(m, max_graph_tokens=a.graph_tokens or (1024 if hybrid else 512))
    app.state.server = srv
    print(f"serving {a.model} on {m.device} {a.host}:{a.port} (temperature {m.head.temperature:.3f})", flush=True)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
