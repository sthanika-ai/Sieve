"""Escaping of caller text for letter-readout models: no request text can produce a special token of the tokenizer (a
turn marker, a tool or media token), exactly as the models were trained.

    Gemma 4: each such token loses its first character to "<¦", e.g. "<|turn>" becomes "<¦|turn>".
    Qwen3.5: "<|...|>" tokens lose their bars, "<|im_start|>" becomes "<¦im_start¦>"; the added tokens of another form
             (<think>, </think>, <tool_call>, <tool_response>, <tts_...>) become "<¦think¦>", "</¦think¦>", ..."""
import re

# trained structural tokens of the Gemma 4 -it model that bracket data (the trainer's Gemma 4 layout)
GEMMA4_SPECIAL = ["<|tool_response>", "<|tool_call>", "<|tool>", "<tool|>", "<tool_call|>"]
GEMMA4_ESCAPE = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
                 "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>", "<|turn>",
                 "<turn|>", "<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>"]
_GEMMA4_RE = re.compile("|".join(re.escape(t) for t in sorted(GEMMA4_ESCAPE, key=len, reverse=True)))

# the trainer's Qwen layout: a Qwen tokenizer has these five tokens (the order of detection is the trainer's: Qwen first)
QWEN_SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_QWEN_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
_QWEN_TAG_RE = re.compile(r"<(/?)(think|tool_call|tool_response|tts_[a-z_]+)>")


def _known(tok, t):
    try:
        i = tok.convert_tokens_to_ids(t)
    except KeyError:
        return False
    return i is not None and i != getattr(tok, "unk_token_id", None)


def escape(text):
    """Gemma 4."""
    return _GEMMA4_RE.sub(lambda m: "<¦" + m.group(0)[1:], text)


def escape_qwen(text):
    """Qwen3.5."""
    return _QWEN_TAG_RE.sub(r"<\1¦\2¦>", _QWEN_SPECIAL_RE.sub(r"<¦\1¦>", text))


def family_of(tok):
    """'qwen' (Qwen3.5) or 'gemma4' (Gemma 4): the letter layout of this tokenizer; any other tokenizer is refused."""
    if all(_known(tok, t) for t in QWEN_SPECIAL):
        return "qwen"
    if all(_known(tok, t) for t in GEMMA4_SPECIAL):
        return "gemma4"
    raise ValueError(f"{type(tok).__name__} is neither a Gemma 4 nor a Qwen3.5 tokenizer: letter-readout models are "
                     "Gemma 4 or Qwen3.5 only")


def escape_of(tok):
    """The escape rule of this tokenizer's family (the trainer's layout)."""
    return escape_qwen if family_of(tok) == "qwen" else escape
