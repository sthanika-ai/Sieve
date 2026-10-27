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
