"""Letter-readout models (sieve/letters, Sieve-31B-v3): family detection, escaping, and the state computed once
(LetterModel.state_once) against the packs, on a tiny random Gemma-4 model on the CPU."""
import json

import pytest
import torch

from sieve.load import family_of


def test_family_letters_and_pointer(tmp_path):
    rel = tmp_path / "release"
    rel.mkdir()
    torch.save({"readout": "letters", "temperature": 1.0}, rel / "head.pt")
    (rel / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    assert family_of(str(rel)) == "letters"
    ad = tmp_path / "adapter"
    ad.mkdir()
    torch.save({"readout": "letters"}, ad / "head.pt")           # an adapter with head.pt is a pointer model, always
    (ad / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    assert family_of(str(ad)) == "pointer"
    other = tmp_path / "other"
    other.mkdir()
    torch.save({"temperature": 1.0}, other / "head.pt")          # no readout: not a letters model
    (other / "config.json").write_text("{}")
    assert family_of(str(other)) == "pointer"


def test_escape():
    from sieve.letters.layout import escape, escape_of
    assert escape("a <|turn>b <bos> <|tool_call> c") == "a <¦|turn>b <¦bos> <¦|tool_call> c"
    assert escape("plain <b> text") == "plain <b> text"

    class NotGemma:
        unk_token_id = 0

        def convert_tokens_to_ids(self, t):
            return 0
    with pytest.raises(ValueError):
        escape_of(NotGemma())


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.gemma4 import modeling_gemma4 as G
    try:
        real = AutoConfig.from_pretrained("google/gemma-4-31B-it").get_text_config()
        tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    except OSError:
        pytest.skip("the Gemma-4-31B-it config and tokenizer are not available")
    torch.manual_seed(0)
    cfg = real.__class__(**{**real.to_dict(), "num_hidden_layers": 6, "hidden_size": 64, "intermediate_size": 128,
                            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "global_head_dim": 32,
                            "num_global_key_value_heads": 1, "vocab_size": 300000, "sliding_window": 8,
                            "layer_types": ["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2
                            + ["full_attention"],
                            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0, "per_layer_config": {}})
    cfg._attn_implementation = "sdpa"
    full = G.Gemma4ForCausalLM(cfg).to(torch.float64).eval()
    return full, tok


REQ = ("Shipment 4471 left the depot on Monday; the customer reports that two boxes arrived crushed and one is "
       "missing. The courier's log shows a scan at the hub on Tuesday and no delivery scan.",
       {"cause": {"type": "choice", "instructions": "What most likely happened to the missing box?",
                  "criteria": {"lost": "lost at the hub", "late": "still in transit", "stolen": "stolen on delivery",
                               "never": "never shipped"}},
        "claim": {"type": "noul", "instructions": "Should the customer get a refund for the crushed boxes?"},
        "who": {"type": "choice", "instructions": "Who should handle the claim?",
                "criteria": {"courier": None, "depot": "the depot", "service": "customer service"}}})


def _logits(m, **policy):
    return m.logits_many([REQ], budget=100000, reads=1, **policy)[0]


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_state_once_matches_packs(tiny, monkeypatch, attention):
    import sieve.letters.attention as A
    from sieve.letters.model import LetterModel
    full, tok = tiny
    m = LetterModel(full, tok, attention, device="cpu", pack_tokens=1)    # pack_tokens=1: every branch its own pack
    st, ins, opts, meta = m.encode(*REQ)
    ns = len(m.fmt.state(ins, st))
    assert ns > m.window and len(ins) == 3
    calls, orig = [], m.state_once
    monkeypatch.setattr(m, "state_once", lambda b0, sets: calls.append(len(sets)) or orig(b0, sets))
    for blocked in (False, True):                                       # one attention call, and query blocks
        monkeypatch.setattr(A, "ATTN_BLOCK", 16 if blocked else 1024)
        monkeypatch.setattr(A, "ATTN_BLOCK_MIN", 16 if blocked else 4096)
        m.PREFIX_MIN = 0                                               # the packs
        calls.clear()
        packs1, packs2 = _logits(m, orders=1), _logits(m, orders=2)
        assert calls == []
        m.PREFIX_MIN, m.BRANCH_TOKENS = 1, 40                           # the state once, branches in several passes
        calls.clear()
        once1, once2 = _logits(m, orders=1), _logits(m, orders=2)
        assert calls == [1, 2]                                          # it ran; the second order as a second set
        m.BRANCH_TOKENS = 100000                                        # ... and all in one pass
        once2_one = _logits(m, orders=2)
        for a, b in ((packs1, once1), (packs2, once2), (packs2, once2_one)):
            assert len(a) == len(b) == 3
            for x, y in zip(a, b):
                assert x.shape == y.shape
                assert torch.allclose(x, y, atol=1e-9, rtol=1e-9), (x, y)
        assert not all(torch.allclose(x, y) for x, y in zip(packs1, packs2))   # the second order changes the answer
    assert not any(getattr(mod, "sieve_capture", None) is not None or getattr(mod, "sieve_prefix", None) is not None
                   for mod in m.lm.modules())                          # nothing left on the layers


def test_state_once_only_where_it_applies(tiny):
    from sieve.letters.model import LetterModel
    full, tok = tiny
    m = LetterModel(full, tok, "bidirectional", device="cpu")
    m.PREFIX_MIN = 50
    assert m.uses_state_once(60, 3, 1, short=False)
    assert not m.uses_state_once(49, 3, 1, short=False)                # a short state stays in the packs
    assert not m.uses_state_once(60, 1, 1, short=False)                # one question: one block, its state reads it
    assert not m.uses_state_once(60, 3, 2, short=True)                 # read twice
    assert m.uses_state_once(60, 3, 2, short=False)                    # too long to read twice: the packs' attention
    m.PREFIX_MIN = 0
    assert not m.uses_state_once(10 ** 6, 3, 1, short=False)           # 0 switches it off


def test_engine_answers_in_typed_format():
    """LettersEngine's answers have every typed-answer field: choice with confidence, noul as P(yes), score via
    to_answers. A stub model returns fixed logits, so no weights are needed."""
    import math
    from sieve.api import choice_confidence, score_confidence
    from sieve.letters.engine import LettersEngine

    class Stub:
        temperature, temperatures = 1.0, None

        def logits_many(self, requests, budget):
            return [[torch.tensor([2.0, 0.5, 0.1, -1.0]), torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 1.0, 3.0])]]

    e = LettersEngine.__new__(LettersEngine)
    e.m, e.budget, e.provenance = Stub(), 16384, {"model": "stub"}
    qs = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None, "c": None, "d": None}},
          "yes": {"type": "noul", "instructions": "Leaving?"},
          "lvl": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}}
    body, _ = e("state", qs)
    a = body["answers"]
    p = torch.softmax(torch.tensor([2.0, 0.5, 0.1, -1.0]), -1).tolist()
    assert a["team"]["choice"] == "a" and math.isclose(a["team"]["confidence"], choice_confidence(p))
    assert set(a["team"]) == {"type", "choice", "confidence", "probabilities"}
    assert set(a["yes"]) == {"type", "noul"} and math.isclose(a["yes"]["noul"], torch.softmax(torch.tensor([-1.0, 1.0]), -1)[1].item())
    q = torch.softmax(torch.tensor([0.0, 1.0, 3.0]), -1).tolist()
    assert set(a["lvl"]) == {"type", "score", "legend", "confidence", "probabilities"}
    assert math.isclose(a["lvl"]["confidence"], score_confidence(q))         # the engine keeps full precision


# ------------------------------------------------------------------------------------------------------------ Qwen3.5
def test_escape_qwen():
    from sieve.letters.layout import escape, escape_of, escape_qwen, family_of
    assert escape_qwen("a <|im_start|>b <think>x</think> <tool_call> <tts_pad> <b>") == \
        "a <¦im_start¦>b <¦think¦>x</¦think¦> <¦tool_call¦> <¦tts_pad¦> <b>"

    class QwenLike:
        unk_token_id = None

        def convert_tokens_to_ids(self, t):
            return 7 if t.startswith(("<|fim_", "<|box_")) else None
    assert family_of(QwenLike()) == "qwen" and escape_of(QwenLike()) is escape_qwen

    class GemmaLike(QwenLike):
        def convert_tokens_to_ids(self, t):
            return 7 if "tool" in t else None
    assert family_of(GemmaLike()) == "gemma4" and escape_of(GemmaLike()) is escape


@pytest.fixture(scope="module")
def qwen_tiny():
    """A tiny random Qwen3.5 text model (linear-attention and full-attention layers alternating) with the real tokenizer.
    A full-attention layer before the last one: only through it can rule A change the answer, read at the last token."""
    from transformers import AutoConfig, AutoTokenizer
    try:
        real = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B").get_text_config()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except OSError:
        pytest.skip("the Qwen3.5-9B config and tokenizer are not available")
    torch.manual_seed(0)
    cfg = real.__class__(**{**real.to_dict(), "num_hidden_layers": 4, "hidden_size": 64, "intermediate_size": 128,
                            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                            "linear_num_key_heads": 2, "linear_num_value_heads": 2, "linear_key_head_dim": 16,
                            "linear_value_head_dim": 16, "full_attention_interval": 2,
                            "layer_types": ["linear_attention", "full_attention"] * 2})
    cfg._attn_implementation = "sdpa"
    # transformers binds the linear-attention kernels (causal-conv1d, fla) when it imports the model module, whatever the
    # device; they are CUDA-only. Re-import the module with them hidden: the PyTorch path, on the CPU in float64.
    import importlib
    import sys
    from transformers.models.qwen3_5 import modeling_qwen3_5
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "causal_conv1d", None)
        mp.setitem(sys.modules, "fla", None)
        M = importlib.reload(modeling_qwen3_5)
    full = M.Qwen3_5ForCausalLM(cfg).to(torch.float64).eval()
    return full, tok


def test_qwen_format(qwen_tiny):
    from sieve.letters.format import QwenLetterFormat, letter_format
    _, tok = qwen_tiny
    f = letter_format(tok)                                   # the labels are checked against the tokenizer here
    assert isinstance(f, QwenLetterFormat)
    assert len(f.label_ids) == len(set(f.label_ids)) == 255 and f.labels[:3] == ["A", "B", "C"]
    st = tok("the state", add_special_tokens=False).input_ids
    ins = [tok("Which one?", add_special_tokens=False).input_ids]
    opts = [tok(x, add_special_tokens=False).input_ids for x in ("red", "blue")]
    text = tok.decode(f.state(ins, st) + f.branch(ins[0], opts))
    assert text.startswith("<|im_start|>system\n") and "\nOption A: red\nOption B: blue\n" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")


def test_qwen_one_row_per_question(qwen_tiny):
    """Several questions: each question is its own row (block 0, then its block), as the trainer runs Qwen3.5; rule A
    in the full-attention layer changes the answer, and one question's options never reach another's answer."""
    from sieve.letters.model import LetterModel
    full, tok = qwen_tiny
    bi = LetterModel(full, tok, "bidirectional", device="cpu")
    ca = LetterModel(full, tok, "causal", device="cpu")
    assert bi.recurrent and not bi.windowed
    st, ins, opts, meta = bi.encode(*REQ)
    units = bi.units(st, ins, opts)
    ns = len(bi.fmt.state(ins, st))
    assert len(units) == 3
    for q, (ids, pos, seg, reads) in enumerate(units):
        b = bi.fmt.branch(ins[q], opts[q])
        assert ids == bi.fmt.state(ins, st) + b and pos == list(range(len(ids)))
        assert seg == [0] * ns + [1] * len(b) and reads == [(q, len(ids) - 1, len(opts[q]))]
    z_bi, z_ca = (m.logits_many([REQ], budget=100000, reads=1, orders=1)[0] for m in (bi, ca))
    assert [z.shape[0] for z in z_bi] == [4, 2, 3] and all(torch.isfinite(z).all() for z in z_bi)
    assert not all(torch.allclose(x, y) for x, y in zip(z_bi, z_ca))            # rule A is applied
    other = (REQ[0], {**REQ[1], "who": {**REQ[1]["who"], "criteria": {"x": None, "y": None, "z": None}}})
    z_other = bi.logits_many([other], budget=100000, reads=1, orders=1)[0]
    assert torch.equal(z_bi[0], z_other[0]) and torch.equal(z_bi[1], z_other[1])   # only "who" changed


@pytest.mark.parametrize("attention", ["bidirectional", "causal"])
def test_qwen_state_once_matches_rows(qwen_tiny, monkeypatch, attention):
    """Qwen3.5 with block 0 run once (the linear-attention states and the attention keys kept, the questions as rows of
    a batch continuing from them) gives each question's own row, block 0 + its block, for both option orders."""
    from sieve.letters.model import LetterModel
    full, tok = qwen_tiny
    m = LetterModel(full, tok, attention, device="cpu")
    calls, orig = [], m.state_once_rows
    monkeypatch.setattr(m, "state_once_rows", lambda b0, sets: calls.append(len(sets)) or orig(b0, sets))
    m.PREFIX_MIN = 0                                                    # one row per question
    rows1, rows2 = _logits(m, orders=1), _logits(m, orders=2)
    assert calls == []
    st, ins, opts, meta = m.encode(*REQ)
    ns = len(m.fmt.state(ins, st))
    for budget in (ns + 1, 10 ** 6):                                    # one row per pass, then all rows in one pass
        m.PREFIX_MIN, m.KEY_TOKENS = 1, budget
        calls.clear()
        once1, once2 = _logits(m, orders=1), _logits(m, orders=2)
        assert calls == [1, 2]
        for a, b in ((rows1, once1), (rows2, once2)):              # transformers' PyTorch linear attention computes
            assert len(a) == len(b) == 3                               # in float32 even in a float64 model, and block 0
            for x, y in zip(a, b):                                     # ends a chunk there: ~1e-8 apart, not 1e-15
                assert x.shape == y.shape and torch.allclose(x, y, atol=1e-6, rtol=0), (x, y)
    assert not any(getattr(mod, "sieve_capture", None) is not None or getattr(mod, "sieve_prefix", None) is not None
                   for mod in m.lm.modules())                          # nothing left on the layers
