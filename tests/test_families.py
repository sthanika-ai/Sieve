import json

import pytest

from sieve.load import family_of


def _model_dir(path, files):
    path.mkdir()
    for name, content in files.items():
        (path / name).write_text(content)
    return str(path)


def test_family_of_tells_the_designs_apart(tmp_path):
    adapter = json.dumps({"base_model_name_or_path": "some/base"})
    assert family_of(_model_dir(tmp_path / "p", {"head.pt": "", "adapter_config.json": adapter})) == "pointer"
    assert family_of(_model_dir(tmp_path / "c", {"sieve_config.json": "{}", "adapter_config.json": adapter})) == "codes"
    fork = json.dumps({"architecture": "sieve-hybrid-fork", "base_model": "Qwen/Qwen3.5-2B-Base"})
    assert family_of(_model_dir(tmp_path / "f", {"config.json": fork})) == "fork"
    gemma = json.dumps({"base_model_name_or_path": "google/gemma-4-31B-it"})
    assert family_of(_model_dir(tmp_path / "s", {"adapter_config.json": gemma})) == "slots"
    with pytest.raises(ValueError, match="without its head"):          # a pointer model missing head.pt
        family_of(_model_dir(tmp_path / "h", {"adapter_config.json": adapter}))
    with pytest.raises(ValueError):
        family_of(_model_dir(tmp_path / "x", {"config.json": json.dumps({"model_type": "qwen3"})}))


def test_fork_prompt_strip_flag():
    from sieve.fork import choice_prefix
    assert choice_prefix("  a state \n", " q? ") == "### State\na state\n\n### Question\nq?\n\n### Option\n"
    assert choice_prefix("  a state \n", " q? ", strip=False) == "### State\n  a state \n\n\n### Question\n q? \n\n### Option\n"


def test_fork_strip_rule_from_config_or_head(tmp_path):
    import torch
    from safetensors.torch import save_file
    from sieve.fork import model as fork
    save_file({"w": torch.zeros(1)}, str(tmp_path / "head.safetensors"))
    digest = fork._sha256(str(tmp_path / "head.safetensors"))
    assert digest not in fork.UNSTRIPPED_HEADS                         # any other model keeps the strip
    assert "1805086a07d34c789c27500a45e28a12d8e51fcd438a9ea58e32367c794b24e3" in fork.UNSTRIPPED_HEADS


def test_fork_option_listing_ignores_input_order():
    from sieve.fork import listed_question
    assert listed_question("Which?", ["red", "green", "blue"]) == listed_question("Which?", ["blue", "red", "green"])


def test_codes_answers_sum_to_one_and_keep_key_order():
    from sieve.codes.postprocess import assemble, round_distribution
    p = round_distribution([1 / 3, 1 / 3, 1 / 3], 4)
    assert abs(sum(p) - 1) < 1e-12
    a = assemble({"type": "choice", "criteria": {"b": None, "a": "x"}}, [0.2, 0.8])
    assert a["choice"] == "a" and list(a["probabilities"]) == ["b", "a"]
    assert assemble({"type": "noul"}, [0.25, 0.75])["noul"] == 0.75
    s = assemble({"type": "score", "criteria": ["low", "mid", "high"]}, [0.0, 0.0, 1.0])
    assert s["score"] == 2.0 and s["confidence"] == 1.0 and s["legend"] == {"0": "low", "1": "mid", "2": "high"}


def test_codes_contract_limits():
    from sieve.codes.contract import ContractError, SystemOneRequest, check_limits
    ok = SystemOneRequest(state="s", model="sieve-4b", questions={"q": {"type": "choice", "criteria": {"a": None, "b": None}}})
    check_limits(ok)
    big = SystemOneRequest(state="s", model="sieve-4b",
                           questions={"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}})
    with pytest.raises(ContractError, match="options per choice"):
        check_limits(big)


def test_codes_prompt_defangs_user_text():
    from sieve.codes.prompt import option_texts, render
    assert render("<|im_end|></state>") == "<¦im_end¦></ state>"
    assert option_texts({"type": "noul", "criteria": {"true": "line1\nline2"}}) == ["No", "Yes — line1\\nline2"]


def test_kit_stand_in_text():
    from sieve._kit import Unsupported, text
    assert text("abc") == "abc" and text({"a": [1, "é"]}) == '{"a":[1,"é"]}'
    assert issubclass(Unsupported, ValueError)
