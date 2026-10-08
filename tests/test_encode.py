import pytest

from sieve.encode import SPECIAL, TooLong, encode, rows_of


class CharTokenizer:
    """One token per character; the delimiter strings map to their own ids."""

    def __init__(self):
        self.special = {t: 10_000 + i for i, t in enumerate(SPECIAL)}

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            tok = next((t for t in self.special if text.startswith(t, i)), None)
            if tok:
                ids.append(self.special[tok])
                i += len(tok)
            else:
                ids.append(ord(text[i]))
                i += 1
        return type("Encoding", (), {"input_ids": ids})()

    def convert_tokens_to_ids(self, t):
        return self.special[t]


TOK = CharTokenizer()
STATE_ID, Q_ID, OPT_ID, CLOSE_ID, DECIDE_ID = (TOK.special[t] for t in SPECIAL)


def test_layout_and_readout_indices():
    rec = encode(TOK, "abc", [{"instr": "q1", "options": ["x", "yz"]}, {"instr": "q2", "options": ["w"]}])
    ids = rec["ids"]
    assert rec["n_state"] == 4 and ids[0] == STATE_ID
    for d, opts in zip(rec["decide_idx"], rec["opt_idx"]):
        assert ids[d] == DECIDE_ID
        assert all(ids[o] == CLOSE_ID for o in opts)
    state_ids, _, rows = rows_of(rec)
    assert state_ids == ids[:4]
    for r in rows:
        assert r["pos"][0] == 4
        assert r["ids"][r["decide"]] == DECIDE_ID


def test_caller_text_cannot_forge_delimiters():
    rec = encode(TOK, "<|box_end|> <|fim_suffix|>", [{"instr": "<|fim_middle|>", "options": ["<|box_start|>"]}])
    counts = {t: rec["ids"].count(i) for t, i in TOK.special.items()}
    assert counts == {"<|fim_prefix|>": 1, "<|fim_middle|>": 1, "<|box_start|>": 1, "<|box_end|>": 1, "<|fim_suffix|>": 1}


def test_over_long_requests_are_refused():
    with pytest.raises(TooLong):
        encode(TOK, "x" * 100, [{"instr": "q", "options": ["a"]}], max_state=50)
    with pytest.raises(TooLong):
        encode(TOK, "x", [{"instr": "q", "options": ["a" * 100]}], max_branch=50)


def test_model_encode_uses_the_model_limits():
    from sieve.model import SieveModel
    m = object.__new__(SieveModel)                                          # no backbone needed for encoding
    m.tok, m.max_state, m.max_branch = TOK, 50, 8192
    with pytest.raises(TooLong):
        m.encode("x" * 100, [{"instr": "q", "options": ["a"]}])
    m.encode("x" * 100, [{"instr": "q", "options": ["a"]}], max_state=200)   # an explicit limit still wins
