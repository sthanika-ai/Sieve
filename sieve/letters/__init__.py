"""Letter-readout Sieve models (Sieve-31B-v3): a full fine-tune of Gemma-4-31B-it whose answer is the model's own
next-token logits of the option labels at "Answer: (", with bidirectional attention in the global layers (rule A).

    format.py     the prompt renderer shared with the trainer
    layout.py     escaping of caller text (Gemma 4 special tokens)
    attention.py  rule A masks and attention for the marked layers
    model.py      LetterModel: loading, passes, the letter readout
    engine.py     LettersEngine for the Decision Index kit
"""


def load(path, **kw):
    """LetterModel of a release folder or Hub repo (see LetterModel.load for the options)."""
    from .engine import resolve
    from .model import LetterModel
    return LetterModel.load(resolve(path), **kw)
