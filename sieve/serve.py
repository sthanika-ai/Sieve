"""HTTP server for any Sieve model.

    python -m sieve.serve --model sthanika-ai/Sieve-27B --graphs --port 8020

POST /v1/decide  {"state": ..., "questions": {...}, "temperature": optional}  ->  typed answers with probabilities
GET  /v1/models  the loaded model and its design; for a pointer-head model also its temperatures, CUDA-graph and
                 prefix-cache state

Pointer-head models (Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3) get the prefix cache and CUDA graphs
(--graphs, --graph-tokens). Every other model is answered through sieve.decide, with its temperature fixed at load time;
--base and --no-merge go to load_sieve.
"""
import argparse
import threading
import time
from collections import OrderedDict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .api import to_answers, to_record, validate_questions
from .load import check_temperature

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
        while len(self.cache) >= self.size:            # make room before the new prefix is allocated
            self.cache.popitem(last=False)
        while True:
            try:
                pre = self.m.prefill_from_record(rec)
                break
            except torch.OutOfMemoryError:
                if not self.cache:                          # nothing left to release: the request itself does not fit
                    raise
            self.cache.popitem(last=False)                  # outside the handler, whose traceback holds the failed pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.cache[key] = pre
        self.misses += 1
        return pre, False

    def _answer(self, rec, t, uncached=False):
        """One attempt at a request: its state (cached or not), then every question. (probabilities, cache hit).
        A single long question runs as one row and reads no state, so its state is not cached."""
        if uncached or self.m.whole_row(rec):
            pre, hit = None, False
        else:
            pre, hit = self.prefix_for(rec)
        if self.graphed is not None:                    # it prefills an uncached state itself, only for a graph
            logits = self.graphed.logits(rec, prefix=pre, temperature=t)
        else:
            logits = self.m.logits(rec, prefix=pre, temperature=t)
        return [torch.softmax(z, -1).cpu() for z in logits], hit

    def decide(self, req: Request):
        try:
            check_temperature(req.temperature)
        except ValueError as e:
            raise HTTPException(422, str(e))
        try:
            validate_questions(req.questions)
            state, qs, meta = to_record(req.state, req.questions)
            rec = self.m.encode(state, qs)
        except ValueError as e:
            raise HTTPException(422, str(e))
        t = req.temperature if req.temperature is not None else self.m.question_temperatures(meta, req.state)
        with self.lock:
            if self.m.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            key = tuple(rec["ids"][:rec["n_state"]])
            uncached = False
            while True:
                try:
                    probs, hit = self._answer(rec, t, uncached)
                    break
                except torch.OutOfMemoryError:
                    if uncached:                            # nothing cached and still too large: it does not fit
                        raise
                # outside the handler, whose traceback held the failed attempt: give back the least recently used
                # other state; when none is left, drop this request's own and try once more without caching it
                others = [k for k in self.cache if k != key]
                if others:
                    self.cache.pop(others[0])
                else:
                    self.cache.pop(key, None)
                    uncached = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if self.m.device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
        return {"answers": to_answers(probs, meta),
                "usage": {"input_tokens": len(rec["ids"]), "state_tokens": rec["n_state"],
                          "questions": len(qs), "output_tokens": 0},
                "latency_ms": round(dt, 1), "prefix_cache_hit": hit}

    def info(self):
        return {"model": self.model_id, "design": "pointer", "base": self.base, "device": str(self.m.device),
                "temperature": self.m.head.temperature,
                "temperatures_by_category": (self.m.temperatures or {}).get("table"),
                "cuda_graphs": None if self.graphed is None else {"captured": len(self.graphed.graphs),
                                                                  "max_graph_tokens": self.graphed.max_graph_tokens},
                "prefix_cache": {"size": self.size, "hits": self.hits, "misses": self.misses,
                                 "cached_states": len(self.cache)}}


class AnyServer:
    """Any other Sieve model, answered through sieve.decide."""

    def __init__(self, model, model_id, family):
        self.m, self.model_id, self.family = model, model_id, family
        self.lock = threading.Lock()

    def decide(self, req: Request):
        from .load import decide as decide_any
        if req.temperature is not None:
            raise HTTPException(422, "this model takes its temperature at load time")
        with self.lock:
            t0 = time.perf_counter()
            try:
                answers = decide_any(self.m, req.state, req.questions)
            except ValueError as e:
                raise HTTPException(422, str(e))
            except Exception as e:
                status = getattr(e, "status", None)
                if status is None:
                    raise
                raise HTTPException(status, getattr(e, "message", str(e)))
            dt = (time.perf_counter() - t0) * 1000
        return {"answers": answers, "usage": {"questions": len(req.questions)}, "latency_ms": round(dt, 1)}

    def info(self):
        return {"model": self.model_id, "design": self.family}


@app.post("/v1/decide")
def decide(req: Request):
    return app.state.server.decide(req)


@app.get("/v1/models")
def models():
    return app.state.server.info()


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

    from .load import family_of, load_pointer, load_sieve
    family = family_of(a.model)
    if family != "pointer":
        if a.graphs or a.graph_tokens:
            print(f"--graphs / --graph-tokens apply to pointer-head models only; ignored for {a.model} ({family})", flush=True)
        app.state.server = AnyServer(load_sieve(a.model, base=a.base, merge=not a.no_merge), a.model, family)
        print(f"serving {a.model} ({family}) on {a.host}:{a.port}", flush=True)
        import uvicorn
        uvicorn.run(app, host=a.host, port=a.port)
        return
    m = load_pointer(a.model, base=a.base, merge=not a.no_merge)
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
