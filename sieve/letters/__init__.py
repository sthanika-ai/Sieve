"""Letter-readout Sieve models: a full fine-tune of a chat model whose answer is the model's own next-token logits of the
option labels at the answer slot, with bidirectional attention in the global layers (rule A). Gemma 4 (Sieve-31B-v3,
slot "Answer: (") and Qwen3.5 (Sieve-9B-v3, slot "Answer:", linear-attention layers causal).

    format.py     the prompt renderer shared with the trainer
    layout.py     escaping of caller text (Gemma 4 and Qwen3.5 special tokens)
    attention.py  rule A masks and attention for the marked layers
    model.py      LetterModel: loading, passes, the letter readout
    engine.py     LettersEngine for the Decision Index kit
"""


def load(path, **kw):
    """LetterModel of a release folder or Hub repo (see LetterModel.load for the options)."""
    from .engine import resolve
    from .model import LetterModel
    return LetterModel.load(resolve(path), **kw)
