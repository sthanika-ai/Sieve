import json

from sieve.load import BASE_REVISIONS, base_of


def test_base_of_reads_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    assert base_of(str(tmp_path)) == ("Qwen/Qwen3.5-9B", BASE_REVISIONS["Qwen/Qwen3.5-9B"])


def test_base_of_unknown_base_is_unpinned(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "some/other-base"}))
    assert base_of(str(tmp_path)) == ("some/other-base", None)
