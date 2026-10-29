"""Sieve-0.8B, Sieve-2B and Sieve-2B-v2: every option is scored as its own continuation of one cached
[state][question] prefix, read by a scalar head at a middle layer of a Qwen3.5 base model.

    from sieve import load_sieve, decide
    m = load_sieve("sthanika-ai/Sieve-2B-v2")
    decide(m, state, questions)               # choice and noul questions
"""
import json

from .model import Sieve, choice_prefix, listed_question

__all__ = ["Sieve", "choice_prefix", "listed_question", "prepare", "answer", "decide"]


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def prepare(m, state, questions):
    """Check a whole request before any of it runs, with the tokenizer only: the input rules every model shares
    (sieve.api.validate_questions), then for each question its option texts (a noul question is a 2-option choice over
    its own false/true descriptions when given, else "No." / "Yes."), no two the same, and the model's context.
    Returns (state text, [(id, type, keys, instructions, texts)]); raises ValueError for a request it refuses."""
    from ..api import validate_questions
    validate_questions(questions)
    state_str = _text(state) if state not in ("", None, {}, []) else ""
    prepared = []
    for k, q in questions.items():
        t = q.get("type")
        if t == "choice":
            keys, crit = list(q["criteria"]), q["criteria"]
        elif t == "noul":
            keys, crit = ["false", "true"], q.get("criteria") or {}
        else:
            raise ValueError(f"question {k!r}: this model answers choice and noul questions, not {t!r}")
        if t == "noul":                             # each side's description, else "No." / "Yes."
            texts = ["No." if crit.get("false") in (None, "") else _text(crit["false"]),
                     "Yes." if crit.get("true") in (None, "") else _text(crit["true"])]
        else:
            texts = [key if crit.get(key) in (None, "") else _text(crit[key]) for key in keys]
        if len(set(texts)) != len(texts):
            raise ValueError(f"question {k!r}: duplicate option text, cannot be scored distinctly")
        instr = _text(q.get("instructions", ""))
        qtext = listed_question(instr, texts) if m.list_options else instr
        pre = m._ids(choice_prefix(state_str, qtext, m.strip))
        if len(pre) + max(len(m._ids(o)) for o in texts) > m.max_len:
            raise ValueError(f"question {k!r}: state + question + option exceed the {m.max_len}-token context")
        prepared.append((k, t, keys, instr, texts))
    return state_str, prepared


def answer(m, prepared):
    """Run the model on prepare()'s output: {id: answer}, in the format of sieve.decide."""
    from ..api import to_answers
    state_str, items = prepared
    probs = [m._probs(state_str, instr, texts) for _, _, _, instr, texts in items]
    return to_answers(probs, [{"id": k, "type": t, "keys": keys} for k, t, keys, _, _ in items])


def decide(m, state, questions):
    """Answer every question about one state: {id: answer}, in the format of sieve.decide. Every question is checked
    (prepare) before any runs, so a refused request (ValueError) costs no model time; nothing is ever trimmed."""
    return answer(m, prepare(m, state, questions))
