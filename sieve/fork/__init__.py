"""Sieve-0.8B, Sieve-2B and Sieve-2B-v2: every option is scored as its own continuation of one cached
[state][question] prefix, read by a scalar head at a middle layer of a Qwen3.5 base model.

    from sieve import load_sieve, decide
    m = load_sieve("sthanika-ai/Sieve-2B-v2")
    decide(m, state, questions)               # choice and noul questions
"""
import json

from .model import Sieve, choice_prefix, listed_question

__all__ = ["Sieve", "choice_prefix", "listed_question", "decide"]


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def decide(m, state, questions):
    """Answer every question about one state: {id: answer}, in the format of sieve.decide. A noul question is a
    2-option choice over its own false/true descriptions when given, else "No." / "Yes.". A request longer than the
    model's context is refused (ValueError), never trimmed."""
    from ..api import to_answers
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object {id: {type, instructions, criteria}}")
    state_str = _text(state) if state not in ("", None, {}, []) else ""
    probs, meta = [], []
    for k, q in questions.items():
        t = q.get("type")
        if t == "choice":
            keys, crit = list(q["criteria"]), q["criteria"]
        elif t == "noul":
            keys, crit = ["false", "true"], q.get("criteria") or {}
        else:
            raise ValueError(f"question {k!r}: this model answers choice and noul questions, not {t!r}")
        if t == "noul" and not crit:
            texts = ["No.", "Yes."]
        else:
            texts = [key if crit.get(key) in (None, "") else _text(crit[key]) for key in keys]
        if len(set(texts)) != len(texts):
            raise ValueError(f"question {k!r}: duplicate option text, cannot be scored distinctly")
        instr = _text(q.get("instructions", ""))
        qtext = listed_question(instr, texts) if m.list_options else instr
        pre = m._ids(choice_prefix(state_str, qtext, m.strip))
        if len(pre) + max(len(m._ids(o)) for o in texts) > m.max_len:
            raise ValueError(f"question {k!r}: state + question + option exceed the {m.max_len}-token context")
        probs.append(m._probs(state_str, instr, texts))
        meta.append({"id": k, "type": t, "keys": keys})
    return to_answers(probs, meta)
