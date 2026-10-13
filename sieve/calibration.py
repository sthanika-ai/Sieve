"""Per-category calibration temperatures.

A question's category is read from the request alone, so it is known at serving time: its type, its number of options
and the kind of state (empty, text, or structured JSON). head.pt may store

    "temperatures": {"rule": "type_options_state", "table": {category: T, ...}, "questions": {category: n, ...}}

next to the global "temperature". A question takes the temperature of its finest category in the table
("choice:3-4:text", then "choice:3-4", then "choice"), else the global one. Models without the table (Sieve-9B,
Sieve-27B) use the global temperature for every question, as before.
"""
RULES = ("type_options_state",)
OPTION_BUCKETS = ((2, "2"), (4, "3-4"), (9, "5-9"), (25, "10-25"))


def option_bucket(k):
    return next((name for hi, name in OPTION_BUCKETS if k <= hi), "26+")


def state_kind(state):
    """empty | text | struct, from the request's state before it is rendered."""
    if state is None or (isinstance(state, str) and not state.strip()) or (isinstance(state, (dict, list)) and not state):
        return "empty"
    return "struct" if isinstance(state, (dict, list)) else "text"


def categories(qtype, n_options, state):
    """The question's categories, finest first."""
    a = f"{qtype}:{option_bucket(n_options)}"
    return [f"{a}:{state_kind(state)}", a, qtype]


def question_temperatures(temps, meta, default, state):
    """One temperature per question of meta (api.to_record), or None when there is no per-category table."""
    if not temps:
        return None
    if temps.get("rule") not in RULES:
        raise ValueError(f"unknown temperature rule {temps.get('rule')!r} in head.pt (this code knows {RULES})")
    table = temps["table"]
    out = []
    for m in meta:
        t = next((table[c] for c in categories(m["type"], len(m["keys"]), state) if c in table), default)
        out.append(float(t))
    return out
