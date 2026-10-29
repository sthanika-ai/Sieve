"""Request -> /v1/systemone response. Pure orchestration around DecisionModel; no HTTP here."""
from __future__ import annotations

import threading
import time

from .contract import SystemOneRequest, check_limits, token_limit_error
from .postprocess import assemble, output_token_count
from .prompt import canonical_probs


class Engine:
    def __init__(self, model, model_name: str = "sieve-4b", decimals: int = 4, noul_both_orders: bool = True,
                 bill_preamble: bool = False, max_batch_tokens: int = 1 << 17, release_reserved_gb: float = 0.0):
        self.model, self.model_name, self.decimals = model, model_name, decimals
        self.noul_both_orders, self.bill_preamble, self.max_batch_tokens = noul_both_orders, bill_preamble, max_batch_tokens
        self.release_reserved = release_reserved_gb * 2**30    # >0: hand cached blocks back when reserved exceeds this
        self.lock = threading.Lock()
        self.preamble_tokens = len(model.prompts._prefix_head)

    def _logits(self, enc):
        """Branch logits, surviving CUDA OOM on a shared GPU: drop the state cache, release cached blocks, retry with
        4x and 16x smaller branch chunks; if even that fails the caller gets 529 Overloaded."""
        import torch
        from .contract import ContractError
        budgets = [self.max_batch_tokens, max(4096, self.max_batch_tokens // 4), max(2048, self.max_batch_tokens // 16)]
        for i, b in enumerate(budgets):
            try:
                return self.model.branch_logits(enc, max_batch_tokens=b)
            except torch.OutOfMemoryError:
                self.model._prefix_cache.clear()
                torch.cuda.empty_cache()
                if i == len(budgets) - 1:
                    raise ContractError(529, "overloaded", "The server is temporarily out of GPU memory. Please retry.")
            finally:
                if self.release_reserved and torch.cuda.is_available() and torch.cuda.memory_reserved() > self.release_reserved:
                    torch.cuda.empty_cache()

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def precheck(self, req: SystemOneRequest):
        """Everything evaluate refuses before the model runs (option / level counts, token limits), with no GPU work:
        raises ContractError, else returns (questions, encoded request, state tokens, tokens per question)."""
        check_limits(req)
        qs = [req.questions[k].model_dump() for k in req.questions]
        enc = self.model.prompts.encode(req.state, qs, noul_both_orders=self.noul_both_orders)
        # token limits: the preamble sits outside them; state counted once; each question once
        state_tokens = len(enc.prefix_ids) - self.preamble_tokens
        per_q = [0] * len(qs)
        for b, qi in zip(enc.branches, enc.owners):
            per_q[qi] = max(per_q[qi], len(b.ids))
        err = token_limit_error(state_tokens + max(per_q), state_tokens + sum(per_q))
        if err is not None:
            raise err
        return qs, enc, state_tokens, per_q

    def evaluate(self, req: SystemOneRequest) -> tuple[dict, dict]:
        qids = list(req.questions)
        qs, enc, state_tokens, per_q = self.precheck(req)
        t0 = time.perf_counter()
        with self.lock:
            logits = self._logits(enc)
        dt = time.perf_counter() - t0
        acc: list[list[float] | None] = [None] * len(qs)
        cnt = [0] * len(qs)
        for b, qi, z in zip(enc.branches, enc.owners, logits):
            p = canonical_probs(b, self.model.probabilities(z, b.qtype))
            acc[qi] = p if acc[qi] is None else [a + x for a, x in zip(acc[qi], p)]
            cnt[qi] += 1
        answers = {qid: assemble(q, [x / cnt[i] for x in acc[i]], self.decimals) for i, (qid, q) in enumerate(zip(qids, qs))}
        usage = {"input_tokens": int(state_tokens + sum(per_q) + (self.preamble_tokens if self.bill_preamble else 0)),
                 "output_tokens": output_token_count(answers, self.count_tokens)}
        body = {"model": self.model_name, "answers": answers, "usage": usage, "assets_used": None}
        return body, {"model_ms": dt * 1000, "branches": len(enc.branches), "prefix_tokens": len(enc.prefix_ids)}
