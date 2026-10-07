"""Request and response format for typed questions: choice, noul and score.

The confidence formulas are modified from Kev (https://github.com/jaredpalmer/kev, Apache-2.0).
"""
import os

MAX_OPTIONS = int(os.environ.get("SIEVE_MAX_OPTIONS", "255"))


def render(v, indent=0):
    """Flatten a string, dict or list into text, keeping field names as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items())


def option_text(name, desc):
    return name if desc in (None, "") else f"{name}: {render(desc)}"


def question_keys(qtype, criteria):
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(state, questions):
    """Validate a request and convert it to (state_text, [{"instr", "options"}], meta).

    Raises ValueError with a message for the caller when the request is malformed.
    """
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object {id: {type, instructions, criteria}}")
    qs, meta = [], []
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"question {qid!r} must be an object with type, instructions and criteria")
        t = q.get("type")
        if t not in ("noul", "choice", "score"):
            raise ValueError(f"question {qid!r}: unknown question type {t!r} (expected choice, noul or score)")
        if q.get("instructions") is None:
            raise ValueError(f"question {qid!r}: instructions are required")
        crit = q.get("criteria")
        if t == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise ValueError(f"question {qid!r}: noul criteria must be an object {{\"true\": ..., \"false\": ...}} or omitted")
            c = crit or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif t == "choice":
            if not isinstance(crit, dict):
                raise ValueError(f"question {qid!r}: choice needs criteria as an object of options {{key: description or null}}")
            if not 1 <= len(crit) <= MAX_OPTIONS:
                raise ValueError(f"question {qid!r}: choice needs 1..{MAX_OPTIONS} options, got {len(crit)}")
            opts = [option_text(k, v) for k, v in crit.items()]
        else:
            if not isinstance(crit, list):
                raise ValueError(f"question {qid!r}: score needs criteria as a list of ordered levels")
            if not 2 <= len(crit) <= MAX_OPTIONS:
                raise ValueError(f"question {qid!r}: score needs 2..{MAX_OPTIONS} levels, got {len(crit)}")
            opts = [render(x) for x in crit]
        m = {"id": qid, "type": t, "keys": question_keys(t, crit)}
        if t == "score":
            m["legend"] = dict(zip(m["keys"], opts))
        qs.append({"instr": render(q["instructions"]), "options": opts})
        meta.append(m)
    return render(state), qs, meta


def choice_confidence(p):
    """(p_max - 1/K) / (1 - 1/K): 0 for a uniform distribution, 1 for a certain one."""
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1 / K) / (1 - 1 / K)


def score_confidence(p):
    """1 - E|level - mode| / (L - 1)."""
    L = len(p)
    mode = max(range(L), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


def to_answers(probs, meta):
    out = {}
    for p, m in zip(probs, meta):
        p = [float(x) for x in p]
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round(p[1], 4)}
        elif m["type"] == "choice":
            out[m["id"]] = {"type": "choice",
                            "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                            "confidence": round(choice_confidence(p), 4),
                            "probabilities": {k: round(v, 4) for k, v in zip(m["keys"], p)}}
        else:
            out[m["id"]] = {"type": "score",
                            "score": round(sum(i * pi for i, pi in enumerate(p)), 4),
                            "legend": m["legend"],
                            "confidence": round(score_confidence(p), 4),
                            "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)}}
    return out
