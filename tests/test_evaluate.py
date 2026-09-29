import math

import numpy as np

from sieve.evaluate import accuracy_ci, gold_index, metrics, summary


def row(logits, label, group="g", type_="choice", src=None):
    return {"logits": np.array(logits, float), "label": label, "group": group, "type": type_, "src": src}


def test_gold_index():
    assert gold_index({"type": "choice", "criteria": {"a": None, "b": None}, "label": "b"}) == 1
    assert gold_index({"type": "noul", "label": True}) == 1
    assert gold_index({"type": "noul", "label": "false"}) == 0
    assert gold_index({"type": "score", "criteria": ["low", "mid", "high"], "label": 2}) == 2


def test_metrics_match_hand_computation():
    rows = [row([2.0, 0.0], 0), row([0.0, 2.0], 0)]
    p = 1 / (1 + math.exp(-2))
    m = metrics(rows, 1.0)
    assert m["accuracy"] == 0.5
    assert abs(m["nll"] - (-math.log(p) - math.log(1 - p)) / 2) < 1e-9
    assert abs(m["brier"] - ((1 - p) ** 2 + p ** 2)) < 1e-9
    assert abs(m["ece"] - (p - 0.5)) < 1e-9
    assert m["wrong_at_p_ge_0.9"] == 0.0
    assert metrics([row([3.0, 0.0], 1)], 1.0)["wrong_at_p_ge_0.9"] == 1.0


def test_temperature_changes_probabilities_not_accuracy():
    rows = [row([3.0, 0.0, 1.0], 0), row([0.0, 1.0, 4.0], 1)]
    a, b = metrics(rows, 1.0), metrics(rows, 2.0)
    assert a["accuracy"] == b["accuracy"] == 0.5
    assert a["nll"] != b["nll"]


def test_accuracy_ci_is_deterministic_and_brackets_the_mean():
    rows = [row([1.0, 0.0], int(i % 3 == 0), group=f"g{i // 2}") for i in range(40)]
    lo, hi = accuracy_ci(rows)
    assert (lo, hi) == tuple(accuracy_ci(rows))
    acc = metrics(rows, 1.0)["accuracy"]
    assert lo <= acc <= hi


def test_summary_breakdowns():
    rows = [row([1.0, 0.0], 0, src="a"), row([0.0, 1.0], 0, src="b", type_="noul")]
    s = summary(rows, 1.5)
    assert s["questions"] == 2 and s["error_rate"] == 0.5
    assert set(s["by_type"]) == {"choice", "noul"} and set(s["by_source"]) == {"a", "b"}
