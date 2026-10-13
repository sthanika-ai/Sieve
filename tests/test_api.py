import pytest

from sieve.api import MAX_OPTIONS, choice_confidence, to_answers, to_record


def test_choice_noul_score_records():
    text, qs, meta = to_record(
        {"subject": "Refund", "body": "Charged twice."},
        {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "sales": None}},
         "angry": {"type": "noul", "instructions": "Is the customer angry?"},
         "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}})
    assert text == "subject: Refund\nbody: Charged twice."
    assert qs[0] == {"instr": "Which team?", "options": ["billing: payments", "sales"]}
    assert qs[1]["options"] == ["no", "yes"]
    assert qs[2]["options"] == ["low", "high"]
    assert [m["keys"] for m in meta] == [["billing", "sales"], ["false", "true"], ["0", "1"]]


def test_empty_instructions_allowed():
    _, qs, _ = to_record("state", {"q": {"type": "noul", "instructions": ""}})
    assert qs[0]["instr"] == ""


@pytest.mark.parametrize("questions", [
    {},
    {"q": "text"},
    {"q": {"type": "text", "instructions": "x"}},
    {"q": {"type": "noul"}},
    {"q": {"type": "choice", "instructions": "x"}},
    {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}},
    {"q": {"type": "score", "instructions": "x"}},
    {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}},
    {"q": {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(MAX_OPTIONS + 1)}}},
])
def test_malformed_requests_are_rejected(questions):
    with pytest.raises(ValueError):
        to_record("state", questions)


def test_answers():
    _, _, meta = to_record("s", {"c": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}},
                                 "n": {"type": "noul", "instructions": "x"},
                                 "s": {"type": "score", "instructions": "x", "criteria": ["lo", "mid", "hi"]}})
    out = to_answers([[0.2, 0.8], [0.3, 0.7], [0.0, 0.5, 0.5]], meta)
    assert out["c"]["choice"] == "b" and out["c"]["probabilities"] == {"a": 0.2, "b": 0.8}
    assert out["n"] == {"type": "noul", "noul": 0.7}
    assert out["s"]["score"] == 1.5 and out["s"]["legend"] == {"0": "lo", "1": "mid", "2": "hi"}


def test_choice_confidence():
    assert choice_confidence([0.5, 0.5]) == 0.0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1.0
