import json

from sieve.load import BASE_REVISIONS, base_of


def test_base_of_reads_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    assert base_of(str(tmp_path)) == ("Qwen/Qwen3.5-9B", BASE_REVISIONS["Qwen/Qwen3.5-9B"])


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
