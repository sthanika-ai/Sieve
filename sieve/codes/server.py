"""FastAPI server for Sieve-4B speaking the /v1/systemone contract.

    python -m sieve.codes.server --ckpt sthanika-ai/Sieve-4B --port 8123 [--device cuda:0] [--name sieve-4b]
    python -m sieve.codes.server --base Qwen/Qwen3.5-4B ...       # zero-shot readout from the base's lm_head rows

--ckpt takes a Hub repo id or a local folder. Env: SIEVE_API_KEY (if set, Bearer auth with 403/401 error bodies),
SIEVE_DECIMALS (4; 2 rounds the wire values to two decimals).
"""
from __future__ import annotations

import argparse
import os
import secrets
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .contract import AUTH_INVALID, AUTH_MISSING, ContractError, SystemOneRequest

RELEASE_DATE = "2026-09-25"


def uuid7_hex() -> str:
    ms = int(time.time() * 1000)
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    v = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | ((rand >> 68) & 0xFFF) << 64 | 0b10 << 62 | (rand & ((1 << 62) - 1))
    return f"{v:032x}"


class _AuthError(Exception):
    def __init__(self, status, body):
        self.status, self.body = status, body


def build_app(engine, aliases: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="sieve-4b", version="0.2.0")
    key = os.environ.get("SIEVE_API_KEY")
    names = aliases or [engine.model_name, "sieve-4b"]

    def auth(request: Request):
        if not key:
            return
        h = request.headers.get("authorization")
        if not h or not h.lower().startswith("bearer ") or not h[7:].strip():
            raise _AuthError(403, AUTH_MISSING)
        if not secrets.compare_digest(h[7:].strip(), key):
            raise _AuthError(401, AUTH_INVALID)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["x-typesafe-request-id"] = "req_" + uuid7_hex()
        resp.headers["server-timing"] = f"total;dur={(time.perf_counter() - t0) * 1000:.1f}"
        return resp

    @app.exception_handler(_AuthError)
    async def _auth_handler(request, exc: _AuthError):
        return JSONResponse(exc.body, status_code=exc.status)

    @app.exception_handler(ContractError)
    async def _contract_handler(request, exc: ContractError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.post("/v1/systemone", dependencies=[Depends(auth)])
    def systemone(req: SystemOneRequest):
        body, stats = engine.evaluate(req)
        return JSONResponse(body, headers={"x-sieve-model-ms": f"{stats['model_ms']:.1f}"})

    @app.get("/v1/models", dependencies=[Depends(auth)])
    def models():
        return {"models": [{"name": n, "description": "General-purpose system one model.", "release_date": RELEASE_DATE}
                           for n in names]}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": engine.model_name}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--base")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--quantization", default=None)
    ap.add_argument("--name", default="sieve-4b"); ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--decimals", type=int, default=int(os.environ.get("SIEVE_DECIMALS", 4)))
    ap.add_argument("--bill-preamble", action="store_true"); ap.add_argument("--noul-one-order", action="store_true")
    ap.add_argument("--prefix-cache-gb", type=float, default=4.0, help="state-prefix LRU cache size (0 disables)")
    ap.add_argument("--max-batch-tokens", type=int, default=1 << 17, help="tokens per branch chunk (prefix + branch each)")
    ap.add_argument("--release-reserved-gb", type=float, default=0.0,
                    help="shared GPU: return cached CUDA blocks to the driver whenever reserved memory exceeds this")
    a = ap.parse_args()
    import uvicorn
    from . import resolve
    from .engine import Engine
    from .model import DecisionModel, ModelConfig
    if a.ckpt:
        model = DecisionModel.load(resolve(a.ckpt), device=a.device, quantization=a.quantization)
    else:
        model = DecisionModel(ModelConfig(base=a.base), device=a.device, quantization=a.quantization)
    model.eval()
    from .model import PrefixCache
    model._prefix_cache = PrefixCache(max_bytes=int(a.prefix_cache_gb * 2**30))
    engine = Engine(model, model_name=a.name, decimals=a.decimals, noul_both_orders=not a.noul_one_order,
                    bill_preamble=a.bill_preamble, max_batch_tokens=a.max_batch_tokens, release_reserved_gb=a.release_reserved_gb)
    uvicorn.run(build_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
