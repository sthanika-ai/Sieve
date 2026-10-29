"""FastAPI server for Sieve-4B speaking the /v1/systemone contract.

    python -m sieve.codes.server --ckpt sthanika-ai/Sieve-4B --port 8123 [--device cuda:0] [--name sieve-4b]
    python -m sieve.codes.server --base Qwen/Qwen3.5-4B ...       # zero-shot readout from the base's lm_head rows

--ckpt takes a Hub repo id or a local folder. Env: SIEVE_API_KEY (if set, Bearer auth with 403/401 error bodies).
A request is read as every Sieve model reads it (a bare number as its text, the input rules of sieve.api) and answered
in the format every Sieve model answers in (sieve.api.standard_answers: 4 decimals, rounded once).
"""
from __future__ import annotations

import argparse
import os
from typing import Any
import secrets
import time

from fastapi import Body, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import ValidationError

from ..api import standard_answers, validate_questions
from . import as_content
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
    from .. import __version__
    app = FastAPI(title="sieve-4b", version=__version__)
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
    def systemone(payload: Any = Body(...)):
        from .postprocess import output_token_count
        questions = payload.get("questions") if isinstance(payload, dict) else None
        if isinstance(questions, dict) and all(isinstance(q, dict) for q in questions.values()):
            state, questions = as_content(payload.get("state"), questions)   # a bare number as its text
            payload = {**payload, "questions": questions, **({"state": state} if "state" in payload else {})}
        try:
            req = SystemOneRequest.model_validate(payload, from_attributes=True)   # as FastAPI validates a body
        except ValidationError as e:                # the schema's own 422 body, located in the request body
            raise RequestValidationError([{**err, "loc": ("body", *err["loc"])} for err in e.errors(include_url=False)])
        try:                                        # the input rules every Sieve model shares, on the questions as sent
            validate_questions(questions)           # (the schema drops keys it does not know, e.g. a noul "yes")
        except ValueError as e:
            raise ContractError(400, "invalid_request", str(e))
        body, stats = engine.evaluate(req)
        answers = standard_answers(body["answers"], questions)       # one answer format, rounded once
        body = {**body, "answers": answers,
                "usage": {**body["usage"], "output_tokens": output_token_count(answers, engine.count_tokens)}}
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
    ap.add_argument("--host", default="127.0.0.1")
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
    engine = Engine(model, model_name=a.name, decimals=None, noul_both_orders=not a.noul_one_order,   # rounded once, above
                    bill_preamble=a.bill_preamble, max_batch_tokens=a.max_batch_tokens, release_reserved_gb=a.release_reserved_gb)
    uvicorn.run(build_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
