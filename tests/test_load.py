import json

from sieve.load import BASE_REVISIONS, base_of


def test_base_of_reads_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    assert base_of(str(tmp_path)) == ("Qwen/Qwen3.5-9B", BASE_REVISIONS["Qwen/Qwen3.5-9B"])


def test_base_of_pins_the_full_fine_tune_backbone(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-2B"}))
    assert base_of(str(tmp_path)) == ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")   # Sieve-2B-v3


def test_base_of_unknown_base_is_unpinned(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "some/other-base"}))
    assert base_of(str(tmp_path)) == ("some/other-base", None)


def test_limits_of_reads_training_config(tmp_path):
    from sieve.load import limits_of
    recipe = {"max_state_tokens": 16384, "max_question_and_options_tokens": 16384}
    (tmp_path / "training_config.json").write_text(json.dumps({"recipe": recipe}))
    assert limits_of(str(tmp_path)) == (16384, 16384)


def test_limits_of_defaults_without_recorded_limits(tmp_path):
    from sieve.load import DEFAULT_LIMITS, limits_of
    assert limits_of(str(tmp_path)) == DEFAULT_LIMITS                      # no training_config.json
    (tmp_path / "training_config.json").write_text(json.dumps({"model": "Sieve-9B"}))
    assert limits_of(str(tmp_path)) == DEFAULT_LIMITS                      # no recipe limits


def _tiny_full_release(tmp_path, lora_value=0.0):
    """A tiny backbone and a release of it as train/to_release.py stores a full fine-tune: every layer a PEFT
    modules_to_save module with changed weights (and a persistent buffer), plus a rank-1 LoRA on the embeddings."""
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)
            self.register_buffer("scale", torch.ones(8))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.layers = nn.ModuleList([Layer(), Layer()])

    torch.manual_seed(0)
    base = Backbone()
    cfg = LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0, target_modules=["embed_tokens"], modules_to_save=["layers.0", "layers.1"])
    pm = get_peft_model(Backbone(), cfg)
    pm.base_model.model.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_embedding" in n:
                p.fill_(lora_value)
        for i in range(2):
            trained = pm.base_model.model.layers[i].modules_to_save["default"]
            trained.proj.weight.add_(1.0 + i); trained.scale.mul_(3.0)
    pm.save_pretrained(str(tmp_path), save_embedding_layers=False)
    return base, Backbone, str(tmp_path)


def test_full_layers_of_tells_a_full_fine_tune_from_an_adapter(tmp_path):
    from sieve.load import full_layers_of
    _, _, full = _tiny_full_release(tmp_path / "full")
    _, _, lora = _tiny_full_release(tmp_path / "lora", lora_value=0.5)     # a LoRA that is not zero: PEFT path
    plain = tmp_path / "plain"; plain.mkdir()
    (plain / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B", "modules_to_save": None}))
    assert full_layers_of(full) is True
    assert full_layers_of(lora) is False
    assert full_layers_of(str(plain)) is False


def test_load_full_layers_builds_the_model_peft_builds(tmp_path):
    import copy
    import torch
    from peft import PeftModel
    from sieve.load import load_full_layers
    base, _, path = _tiny_full_release(tmp_path)
    ref = PeftModel.from_pretrained(copy.deepcopy(base), path).merge_and_unload().state_dict()
    load_full_layers(base, path)
    got = base.state_dict()
    assert set(got) == set(ref) and all(torch.equal(got[k], ref[k]) for k in ref)
    assert torch.equal(got["layers.1.scale"], torch.full((8,), 3.0))      # persistent buffers are loaded too


def test_load_full_layers_refuses_an_incomplete_release(tmp_path):
    import pytest
    from safetensors.torch import load_file, save_file
    from sieve.load import load_full_layers
    base, _, path = _tiny_full_release(tmp_path)
    sd = load_file(f"{path}/adapter_model.safetensors")
    save_file({k: v for k, v in sd.items() if "layers.1.proj.bias" not in k}, f"{path}/adapter_model.safetensors")
    with pytest.raises(ValueError, match="missing"):
        load_full_layers(base, path)
