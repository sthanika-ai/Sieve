"""Token layout: the state once, then one branch per question.

    <state> state tokens
    <q> instructions <opt> option_1 </opt> ... <opt> option_K </opt> <decide>

Every branch's positions continue from the end of the state. The pointer head reads the hidden state at each
</opt> and at <decide>.
"""
import re

# <state>, <q>, <opt>, </opt>, <decide>
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class TooLong(ValueError):
    """The request exceeds the declared token limits. It is refused, never truncated."""


def user_tokens(tok, text):
    """Tokenize caller text so that it cannot produce a delimiter token."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def special_ids(tok):
    return [tok.convert_tokens_to_ids(t) for t in SPECIAL]


def encode(tok, state, questions, max_state=8192, max_branch=8192):
    """questions: [{"instr": str, "options": [str, ...]}]. Returns token ids, positions and readout indices."""
    st_id, q_id, o_id, c_id, d_id = special_ids(tok)
    st = user_tokens(tok, state)
    if len(st) + 1 > max_state:
        raise TooLong(f"state is {len(st) + 1} tokens (max {max_state})")
    S = [st_id] + st
    ids, pos = list(S), list(range(len(S)))
    decide_idx, opt_idx = [], []

    for k, q in enumerate(questions, start=1):
        head = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        branch = head + [t for sp in spans for t in sp] + [d_id]
        if len(branch) > max_branch:
            raise TooLong(f"question {k} branch is {len(branch)} tokens (max {max_branch})")
        base, p0 = len(ids), len(S)
        ends, cursor = [], len(head)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += branch
        pos += list(range(p0, p0 + len(branch)))
        decide_idx.append(base + len(branch) - 1)
        opt_idx.append([base + e for e in ends])

    return {"ids": ids, "pos": pos, "n_state": len(S), "decide_idx": decide_idx, "opt_idx": opt_idx}


def rows_of(rec):
    """Split an encoded record into (state_ids, state_positions, one row per question)."""
    Ls = rec["n_state"]
    rows, start = [], Ls
    for d, oi in zip(rec["decide_idx"], rec["opt_idx"]):
        end = d + 1
        rows.append({"ids": rec["ids"][start:end], "pos": rec["pos"][start:end],
                     "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return rec["ids"][:Ls], rec["pos"][:Ls], rows
