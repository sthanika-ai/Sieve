"""Sieve-31B-v2's attention (sieve/slots/readout.py): non-causal global layers, causal sliding layers, on a tiny random
Gemma-4 text model on the CPU."""
import json

import pytest
import torch

pytest.importorskip("decider")
from sieve.slots.readout import attention_of, bidir_global_masks, set_global_noncausal  # noqa: E402


def test_attention_of_reads_training_config(tmp_path):
    assert attention_of(str(tmp_path)) == "causal"                                     # no training_config.json
    (tmp_path / "training_config.json").write_text(json.dumps({"model": "Sieve-31B"}))
    assert attention_of(str(tmp_path)) == "causal"                                     # Sieve-31B records none
    (tmp_path / "training_config.json").write_text(json.dumps({"attention": "bidir_global"}))
    assert attention_of(str(tmp_path)) == "bidir_global"
    (tmp_path / "training_config.json").write_text(json.dumps({"attention": "sideways"}))
    with pytest.raises(ValueError):
        attention_of(str(tmp_path))


def test_masks_unpadded_and_padded():
    att = torch.ones(1, 6, dtype=torch.long)
    m = bidir_global_masks(att, window=3)
    assert m["full_attention"] is None                                   # fused non-causal kernel, no mask
    s = m["sliding_attention"][0, 0]
    assert s[5].tolist() == [False, False, False, True, True, True]      # causal, last 3 positions
    att = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    m = bidir_global_masks(att, window=8)
    f = m["full_attention"]
    assert f.shape == (2, 1, 4, 4)
    assert f[1, 0, 0].tolist() == [True, True, False, False]             # a query sees later tokens, never padding
    assert m["sliding_attention"][1, 0, 1].tolist() == [True, True, False, False]


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig
    from transformers.models.gemma4 import modeling_gemma4 as G
    try:
        real = AutoConfig.from_pretrained("google/gemma-4-31B-it").get_text_config()
    except OSError:
        pytest.skip("the Gemma-4-31B-it config is not available")
    torch.manual_seed(0)
    cfg = real.__class__(**{**real.to_dict(), "num_hidden_layers": 6, "hidden_size": 64, "intermediate_size": 128,
                            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "global_head_dim": 32,
                            "num_global_key_value_heads": 1, "vocab_size": 300000, "sliding_window": 8,
                            "layer_types": ["sliding_attention"] * 2 + ["full_attention"] + ["sliding_attention"] * 2
                            + ["full_attention"],
                            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0, "per_layer_config": {}})
    cfg._attn_implementation = "sdpa"
    m = G.Gemma4TextModel(cfg).eval()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.config = m, type("C", (), {"get_text_config": lambda self=None: cfg})()
    return m, Wrap(), cfg


def test_bidir_global_on_a_tiny_gemma(tiny):
    m, wrap, cfg = tiny
    ids = torch.randint(1000, 5000, (2, 20))
    att = torch.ones(2, 20, dtype=torch.long)
    att[1, 13:] = 0
    with torch.no_grad():
        causal = m(input_ids=ids, attention_mask=att).last_hidden_state
        assert set_global_noncausal(wrap) == 2
        run = lambda i, a: m(input_ids=i, attention_mask=bidir_global_masks(a, cfg.sliding_window)).last_hidden_state
        both = run(ids, att)
        alone = run(ids[1:, :13], att[1:, :13])                          # unpadded: no mask on the global layers
        first = run(ids[:1], att[:1])
    assert not torch.allclose(both[0], causal[0], atol=1e-4)             # the global layers now look ahead
    assert torch.allclose(alone[0], both[1, :13], atol=1e-4, rtol=1e-4)  # padding never leaks into the real tokens
    assert torch.allclose(first[0], both[0], atol=1e-4, rtol=1e-4)       # rows of a batch stay independent
