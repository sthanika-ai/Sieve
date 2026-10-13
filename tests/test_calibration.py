import pytest

from sieve.api import to_record
from sieve.calibration import categories, option_bucket, question_temperatures, state_kind
from sieve.evaluate import metrics, set_temperatures


def test_categories_finest_first():
    assert categories("choice", 3, "text") == ["choice:3-4:text", "choice:3-4", "choice"]
    assert categories("noul", 2, {"a": 1}) == ["noul:2:struct", "noul:2", "noul"]
    assert [option_bucket(k) for k in (2, 3, 4, 5, 9, 10, 25, 26, 255)] == ["2", "3-4", "3-4", "5-9", "5-9", "10-25", "10-25", "26+", "26+"]
    assert [state_kind(s) for s in ("", "  ", None, {}, [], "x", {"a": 1}, [1], 3)] == \
        ["empty", "empty", "empty", "empty", "empty", "text", "struct", "struct", "text"]


def test_question_temperatures_fall_back_to_coarser_then_global():
    _, _, meta = to_record("state", {"a": {"type": "choice", "instructions": "x", "criteria": {"p": None, "q": None, "r": None}},
                                     "b": {"type": "noul", "instructions": "x"},
                                     "c": {"type": "score", "instructions": "x", "criteria": ["lo", "hi"]}})
    temps = {"rule": "type_options_state", "table": {"choice:3-4:text": 0.8, "noul": 1.3}}
    assert question_temperatures(temps, meta, 1.1, "state") == [0.8, 1.3, 1.1]
    assert question_temperatures(temps, meta, 1.1, "") == [1.1, 1.3, 1.1]
    assert question_temperatures(None, meta, 1.1, "state") is None


def test_unknown_rule_is_refused():
    _, _, meta = to_record("s", {"b": {"type": "noul", "instructions": "x"}})
    with pytest.raises(ValueError):
        question_temperatures({"rule": "type", "table": {}}, meta, 1.0, "s")


def test_evaluation_uses_each_rows_category_temperature():
    rows = [{"logits": [2.0, 0.0], "label": 0, "type": "noul", "cats": categories("noul", 2, "s")},
            {"logits": [2.0, 0.0, 0.0], "label": 0, "type": "choice", "cats": categories("choice", 3, {"a": 1})}]
    set_temperatures(rows, 1.0, {"noul:2": 2.0, "choice:3-4:struct": 0.5})
    assert [r["t"] for r in rows] == [2.0, 0.5]
    per_category = metrics(rows, None)["nll"]
    assert per_category != metrics(rows, 1.0)["nll"]
    assert metrics(set_temperatures(rows, 1.0, None), None)["nll"] == pytest.approx(metrics(rows, 1.0)["nll"])
