"""Request and response format for typed questions: choice, noul and score.

The confidence formulas are modified from Kev (https://github.com/jaredpalmer/kev, Apache-2.0).
"""
import os

MAX_OPTIONS = int(os.environ.get("SIEVE_MAX_OPTIONS", "255"))
MAX_SCORE_LEVELS = 10


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


def validate_questions(questions):
    """The input rules every model answers by (sieve.decide, sieve-serve, the model Spaces): one question object per id;
    every question has non-empty instructions; choice: criteria an object of 2..MAX_OPTIONS options with non-empty text
    keys; score: criteria a
    list of 2..MAX_SCORE_LEVELS distinct levels with text, lowest first; noul: criteria left out or {"true": ..., "false": ...}.
    Raises ValueError with a message for the caller. (The Decision Index engines check only what their models need.)"""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object {id: {type, instructions, criteria}}")
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"question {qid!r} must be an object with type, instructions and criteria")
        t = q.get("type")
        if t not in ("noul", "choice", "score"):
            raise ValueError(f"question {qid!r}: unknown question type {t!r} (expected choice, noul or score)")
        if not render(q.get("instructions")).strip():
            raise ValueError(f"question {qid!r}: instructions are required")
        crit = q.get("criteria")
        if t == "choice":
            if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_OPTIONS:
                raise ValueError(f"question {qid!r}: choice needs criteria as an object of 2..{MAX_OPTIONS} options "
                                 "{key: description or null}")
            if not all(isinstance(k, str) and k.strip() for k in crit):
                raise ValueError(f"question {qid!r}: every choice option needs a non-empty text key")
        elif t == "score":
            if not isinstance(crit, list) or not 2 <= len(crit) <= MAX_SCORE_LEVELS:
                raise ValueError(f"question {qid!r}: score needs criteria as a list of 2..{MAX_SCORE_LEVELS} levels, "
                                 "lowest first")
            if any(not render(x).strip() for x in crit):
                raise ValueError(f"question {qid!r}: every score level needs text")
            if len({render(x) for x in crit}) != len(crit):
                raise ValueError(f"question {qid!r}: two score levels are the same")
        elif crit is not None and (not isinstance(crit, dict) or set(crit) - {"true", "false"}):
            raise ValueError(f"question {qid!r}: noul criteria must be {{\"true\": ..., \"false\": ...}} or left out")


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
    """1 - E|level - mode| / D, at least 0, where D = mean over levels i of |i - (L - 1) / 2| is that distance for a
    uniform distribution: 0 for a uniform distribution (or one split between the ends of the scale), 1 for a certain
    one. With two levels it equals choice_confidence."""
    L = len(p)
    if L == 1:
        return 1.0
    s = sum(p)
    p = [x / s for x in p] if s > 0 else [1 / L] * L
    mode = max(range(L), key=lambda i: p[i])
    D = sum(abs(i - (L - 1) / 2) for i in range(L)) / L
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D)


def to_answers(probs, meta, decimals=4):
    """{id: answer} from each question's probabilities, rounded to `decimals` (None: full precision, as the Decision
    Index engines keep it)."""
    r = (lambda x: round(x, decimals)) if decimals is not None else float
    out = {}
    for p, m in zip(probs, meta):
        p = [float(x) for x in p]
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": r(p[1])}
        elif m["type"] == "choice":
            out[m["id"]] = {"type": "choice",
                            "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])],
                            "confidence": r(choice_confidence(p)),
                            "probabilities": {k: r(v) for k, v in zip(m["keys"], p)}}
        else:
            out[m["id"]] = {"type": "score",
                            "score": r(sum(i * pi for i, pi in enumerate(p))),
                            "legend": m["legend"],
                            "confidence": r(score_confidence(p)),
                            "probabilities": {str(i): r(v) for i, v in enumerate(p)}}
    return out


def standard_answers(answers, questions):
    """Any engine's answers in to_answers' shape: the same fields in the same order, numbers rounded to 4 decimals,
    a score's legend rendered from its question's levels. Engines that add fields (Sieve-4B's "stats") or keep full
    precision (sieve.letters) then answer alike. Idempotent on to_answers' output. (The answer-slot models are
    answered from their unrounded probabilities instead: sieve.slots.engine.SlotsEngine.answers.)"""
    out = {}
    for qid, a in answers.items():
        t = a["type"]
        if t == "noul":
            out[qid] = {"type": "noul", "noul": round(float(a["noul"]), 4)}
            continue
        p = {k: float(v) for k, v in a["probabilities"].items()}
        if t == "choice":
            conf = float(a["confidence"]) if "confidence" in a else choice_confidence(list(p.values()))
            out[qid] = {"type": "choice", "choice": a["choice"], "confidence": round(conf, 4),
                        "probabilities": {k: round(v, 4) for k, v in p.items()}}
        else:
            levels = [p[str(i)] for i in range(len(p))]
            score = float(a["score"])
            conf = float(a["confidence"]) if "confidence" in a else score_confidence(levels)
            out[qid] = {"type": "score", "score": round(score, 4),
                        "legend": {str(i): render(x) for i, x in enumerate(questions[qid]["criteria"])},
                        "confidence": round(conf, 4), "probabilities": {str(i): round(v, 4) for i, v in enumerate(levels)}}
    return out
