"""Escaping of caller text for Gemma 4 letter-readout models: no request text can produce a special token of the
tokenizer (a turn marker, a tool or media token). Each such token in the text loses its first character to "<¦", e.g.
"<|turn>" becomes "<¦|turn>", exactly as Sieve-31B-v3 was trained."""
import re

# trained structural tokens of the Gemma 4 -it model that bracket data (the trainer's Gemma 4 layout)
GEMMA4_SPECIAL = ["<|tool_response>", "<|tool_call>", "<|tool>", "<tool|>", "<tool_call|>"]
GEMMA4_ESCAPE = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
                 "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>", "<|turn>",
                 "<turn|>", "<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>"]
_GEMMA4_RE = re.compile("|".join(re.escape(t) for t in sorted(GEMMA4_ESCAPE, key=len, reverse=True)))


def _known(tok, t):
    try:
        i = tok.convert_tokens_to_ids(t)
    except KeyError:
        return False
    return i is not None and i != getattr(tok, "unk_token_id", None)


def escape(text):
    return _GEMMA4_RE.sub(lambda m: "<¦" + m.group(0)[1:], text)


def escape_of(tok):
    """The escape rule of a Gemma 4 tokenizer (the trainer's Gemma 4 layout); any other tokenizer is refused."""
    if not all(_known(tok, t) for t in GEMMA4_SPECIAL):
        raise ValueError(f"{type(tok).__name__} is not a Gemma 4 tokenizer: letter-readout models are Gemma 4 only")
    return escape
