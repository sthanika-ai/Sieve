"""Answer assembly: probabilities -> /v1/systemone answers with their confidences.

Confidences are computed on the unrounded distribution and then rounded.
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence


def normalize(p: Sequence[float]) -> list[float]:
    p = [max(0.0, float(x)) if math.isfinite(float(x)) else 0.0 for x in p]
    s = sum(p)
    return [1.0 / len(p)] * len(p) if s <= 0 else [x / s for x in p]


def choice_confidence(p: Sequence[float]) -> float:
    if len(p) == 1:
        return 1.0
    p = normalize(p)
    u = 1.0 / len(p)
    return max(0.0, min(1.0, (max(p) - u) / (1.0 - u)))


def score_confidence(p: Sequence[float]) -> float:
    if len(p) == 1:
        return 1.0
    p = normalize(p)
    L = len(p)
    mode = max(range(L), key=p.__getitem__)       # first max
    d = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (L - 1) / 2
    mad_uniform = sum(abs(i - c) for i in range(L)) / L
    return max(0.0, 1.0 - d / mad_uniform)


def round_distribution(p: Sequence[float], decimals: int) -> list[float]:
    """Round to `decimals` and push the rounding residue onto the largest entry so the sum stays 1."""
    q = [round(x, decimals) for x in p]
    resid = round(1.0 - sum(q), decimals)
    if resid != 0 and q:
        i = max(range(len(q)), key=q.__getitem__)
        q[i] = round(min(1.0, max(0.0, q[i] + resid)), decimals)
    return [float(x) for x in q]


def assemble(question: Any, probs: Sequence[float], decimals: int = 4) -> dict:
    """question: a contract Question (pydantic) or dict with the same fields; probs: over the question's options in
    serialization order (choice: criteria order; score: level order; noul: [P(no), P(yes)])."""
    qtype = question.type if hasattr(question, "type") else question["type"]
    p = normalize(probs)
    if qtype == "noul":
        return {"type": "noul", "noul": round(p[1], decimals), "stats": {}}
    if qtype == "choice":
        crit = question.criteria if hasattr(question, "criteria") else question["criteria"]
        keys = list(crit)
        best = max(range(len(p)), key=p.__getitem__)       # ties -> first in criteria order
        conf = choice_confidence(p)
        return {"type": "choice", "choice": keys[best], "confidence": round(conf, decimals),
                "probabilities": dict(zip(keys, round_distribution(p, decimals))), "stats": {}}
    levels = question.criteria if hasattr(question, "criteria") else question["criteria"]
    score = sum(i * pi for i, pi in enumerate(p))
    return {"type": "score", "score": round(score, decimals), "confidence": round(score_confidence(p), decimals),
            "legend": {str(i): lv for i, lv in enumerate(levels)},
            "probabilities": {str(i): v for i, v in enumerate(round_distribution(p, decimals))}, "stats": {}}


def output_token_count(answers: dict, count_tokens) -> int:
    """`output_tokens` is the token count of the serialized answers (it moves with the question id, which the model
    never sees): the compact JSON of the answers object."""
    return int(count_tokens(json.dumps(answers, separators=(",", ":"), ensure_ascii=False)))
