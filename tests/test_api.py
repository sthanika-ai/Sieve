import pytest

from sieve.api import MAX_OPTIONS, choice_confidence, score_confidence, to_answers, to_record


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


def test_score_confidence():
    assert score_confidence([0.2] * 5) == 0.0                      # no preference
    assert score_confidence([0.5, 0, 0, 0, 0.5]) == 0.0            # split between the ends of the scale
    assert abs(score_confidence([0.5, 0.5, 0, 0, 0]) - 7 / 12) < 1e-12   # split between neighbours
    assert score_confidence([0, 1.0, 0]) == 1.0
    for p in ([0.3, 0.7], [0.9, 0.1], [0.5, 0.5]):                 # two levels: the same as a choice
        assert abs(score_confidence(p) - choice_confidence(p)) < 1e-12
    # a reference answer: confidence computed on unrounded probabilities, the probabilities shown rounded to 2 decimals
    assert abs(score_confidence([0.01, 0.06, 0.93]) - 0.89) < 0.015


QS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"delivery": None, "kitchen": None}},
      "leave": {"type": "noul", "instructions": "Leaving?"},
      "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "angry", "furious"]}}


def test_standard_answers_one_shape_for_every_engine():
    from sieve.api import standard_answers
    _, _, meta = to_record("s", QS)
    ref = to_answers([[0.82233, 0.17767], [0.06914, 0.93086], [0.06222, 0.56143, 0.37635]], meta)
    assert standard_answers(ref, QS) == ref                                  # to_answers' output is already standard
    assert [list(a) for a in standard_answers(ref, QS).values()] == [list(a) for a in ref.values()]
    full = {"team": {"type": "choice", "choice": "delivery", "confidence": 0.6446600000001,      # full precision
                     "probabilities": {"delivery": 0.82233000001, "kitchen": 0.17766999999}},
            "leave": {"type": "noul", "noul": 0.930860000002},
            "anger": dict(ref["anger"])}
    assert standard_answers(full, QS) == ref
    extra = {"team": {**ref["team"], "stats": {}}, "leave": {**ref["leave"], "stats": {}},       # added fields
             "anger": {**ref["anger"], "stats": {}}}
    assert standard_answers(extra, QS) == ref


def test_decision_index_engine_answers_score_as_score(monkeypatch):
    import sys
    import types
    import torch
    if "decision_index" not in sys.modules:                                  # the kit is optional: stub its two names
        kit = types.ModuleType("decision_index"); engines = types.ModuleType("decision_index.engines")
        engines.Engine, engines.Unsupported = object, Exception
        kit.engines = engines
        monkeypatch.setitem(sys.modules, "decision_index", kit)
        monkeypatch.setitem(sys.modules, "decision_index.engines", engines)
    from sieve.decision_index_engine import SieveEngine
    e = object.__new__(SieveEngine)
    e.T, e.provenance = 1.0, {"model": "stub"}
    e.max_state = e.max_branch = None
    e.m = types.SimpleNamespace(encode=lambda *a, **k: {"ids": [0]}, question_temperatures=lambda meta, state: None)
    e._logits = lambda text, qs, rec, one_at_a_time=False: [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
                                                            torch.tensor([0.0, 1.0, 2.0])]
    body, _ = e("s", QS)
    a = body["answers"]
    assert a["anger"]["type"] == "score" and set(a["anger"]) == {"type", "score", "legend", "confidence", "probabilities"}
    assert a["team"]["type"] == "choice" and "confidence" in a["team"] and a["leave"]["type"] == "noul"


def test_validate_questions_same_rules_for_every_model():
    from sieve.api import MAX_SCORE_LEVELS, validate_questions
    ok = [{"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": "described"}},
          {"type": "noul", "instructions": "Spam?"},
          {"type": "noul", "instructions": "Spam?", "criteria": {"true": "is spam"}},
          {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(MAX_SCORE_LEVELS)]}]
    for q in ok:
        validate_questions({"q": q})
    bad = [{"type": "choice", "instructions": "Pick", "criteria": {"a": None}},                  # 1 option
           {"type": "choice", "criteria": {"a": None, "b": None}},                                # no instructions
           {"type": "noul", "instructions": "  ", "criteria": {"true": "is spam"}},               # blank instructions
           {"type": "score", "instructions": "Rate", "criteria": ["low"]},                        # 1 level
           {"type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(MAX_SCORE_LEVELS + 1)]},
           {"type": "score", "instructions": "Rate", "criteria": ["low", "low"]},                 # repeated level
           {"type": "score", "instructions": "Rate", "criteria": {"0": "low", "1": "high"}},      # levels as an object
           {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]},                    # options as a list
           {"type": "noul", "instructions": "Spam?", "criteria": {"yes": "is spam"}},             # unknown noul key
           {"type": "rank", "instructions": "Order"}]
    for q in bad:
        with pytest.raises(ValueError):
            validate_questions({"q": q})
    with pytest.raises(ValueError):
        validate_questions({})


def test_decide_applies_the_rules_and_refuses_with_valueerror(monkeypatch):
    import sieve.codes as codes
    from sieve import decide
    from sieve.codes.contract import ContractError
    from sieve.codes.engine import Engine as CodesServing
    from sieve.letters.engine import LettersEngine

    class Never:
        def logits_many(self, *a, **k):
            raise AssertionError("an invalid request must be refused before the model runs")
    e = LettersEngine.__new__(LettersEngine)
    e.m, e.budget, e.provenance = Never(), 1, {"model": "x"}
    with pytest.raises(ValueError, match="instructions are required"):
        decide(e, "s", {"q": {"type": "choice", "criteria": {"a": None, "b": None}}})
    c = CodesServing.__new__(CodesServing)
    def refuse(*a, **k):
        raise ContractError(400, "invalid_request", "over the token limit")
    monkeypatch.setattr(codes, "decide", refuse)
    with pytest.raises(ValueError, match="over the token limit"):
        decide(c, "s", {"q": {"type": "noul", "instructions": "Spam?"}})


def test_fork_choice_reports_probability_and_confidence():
    import numpy as np
    from sieve.fork import Sieve
    s = Sieve.__new__(Sieve)
    s._schemas = {}
    s._probs = lambda state, question, options: np.array([0.83, 0.12, 0.05])
    out = s.choice("state", "Which team?", ["billing", "tech", "sales"])
    assert out["decision"] == "billing" and out["index"] == 0
    assert abs(out["probability"] - 0.83) < 1e-9 and abs(out["confidence"] - choice_confidence([0.83, 0.12, 0.05])) < 1e-9


def test_more_input_rules_and_temperatures():
    from sieve.api import validate_questions
    from sieve.load import check_temperature, load_pointer, load_sieve
    for crit in ({1: None, 2: None}, {"": None, "x": None}, {"  ": None, "x": None}):
        with pytest.raises(ValueError):
            validate_questions({"q": {"type": "choice", "instructions": "Pick", "criteria": crit}})
    for levels in ([None, "today"], ["", "today"], ["  ", "today"]):                # every level needs text
        with pytest.raises(ValueError, match="needs text"):
            validate_questions({"q": {"type": "score", "instructions": "Urgency?", "criteria": levels}})
    for t in (0, -1.0, 0.0, float("inf"), float("nan"), 1e-50, 5e-4):      # 1e-50 underflows in float32
        with pytest.raises(ValueError):
            check_temperature(t)
        with pytest.raises(ValueError):                                     # refused before anything is loaded
            load_sieve("no/such-model", temperature=t)
        with pytest.raises(ValueError):
            load_pointer("no/such-model", temperature=t)
    check_temperature(None); check_temperature(1.0); check_temperature(1e-3)
    import torch
    from fastapi import HTTPException
    from sieve.model import PointerHead
    from sieve.serve import Request, Server
    with pytest.raises(HTTPException) as e:                                 # sieve-serve: 422, not a 500 from NaN
        Server.__new__(Server).decide(Request(state="s", questions={"q": {"type": "choice", "instructions": "Pick",
                                              "criteria": {"a": None, "b": None}}}, temperature=1e-50))
    assert e.value.status_code == 422
    head = PointerHead(4, dp=4).eval()                                      # the smallest accepted one stays finite
    z = head(torch.randn(4), torch.randn(3, 4) * 100, temperature=1e-3)
    assert torch.isfinite(torch.softmax(z, -1)).all()


def test_eval_labels():
    from sieve.evaluate import gold_index
    levels = {"type": "score", "criteria": ["a", "b", "c"]}
    assert gold_index({**levels, "label": 2}) == 2
    for bad in (3, -1, True, "x", 1.5, "3"):
        with pytest.raises(ValueError):
            gold_index({**levels, "label": bad})
    assert [gold_index({**levels, "label": x}) for x in (2, 2.0, "2", " 1 ")] == [2, 2, 2, 1]   # whole numbers
    assert [gold_index({"type": "noul", "label": x}) for x in (True, "true", 1, False, "false", 0)] == [1, 1, 1, 0, 0, 0]
    for bad in ("True", "yes", 2):
        with pytest.raises(ValueError):
            gold_index({"type": "noul", "label": bad})


def test_decide_refusals_prefix_and_retryable_errors(monkeypatch):
    import sieve.codes as codes
    from sieve import decide
    from sieve.codes.contract import ContractError
    from sieve.codes.engine import Engine as CodesServing
    c = CodesServing.__new__(CodesServing)
    q = {"q": {"type": "noul", "instructions": "Spam?"}}
    def overloaded(*a, **k):
        raise ContractError(529, "overloaded", "out of GPU memory, retry")
    monkeypatch.setattr(codes, "decide", overloaded)
    with pytest.raises(ContractError):                                     # not the request's fault: not a ValueError
        decide(c, "s", q)
    with pytest.raises(ValueError, match="prefix"):
        decide(c, "s", q, prefix=object())


def test_sieve_4b_takes_bare_numbers():
    from sieve.codes import as_content
    from sieve.codes.contract import SystemOneRequest
    state, qs = as_content(42, {"a": {"type": "score", "instructions": 7, "criteria": [1, 2, 3]},
                                "b": {"type": "choice", "instructions": "Pick", "criteria": {"x": 1, "y": None}}})
    assert state == "42" and qs["a"]["criteria"] == ["1", "2", "3"] and qs["b"]["criteria"] == {"x": "1", "y": None}
    SystemOneRequest(state=state, model="m", questions=qs)


def test_fork_noul_options_default_per_side():
    import types
    from sieve.fork import decide as decide_fork
    from sieve.fork.engine import ForkEngine
    seen = []
    m = types.SimpleNamespace(list_options=False, strip=False, max_len=10 ** 6, _ids=lambda s: [0],
                              _probs=lambda state, instr, texts: seen.append(texts) or [0.5] * len(texts))
    for crit in (None, {}, {"true": None, "false": None}, {"true": "threatens to cancel"}):
        q = {"type": "noul", "instructions": "Leaving?"} | ({} if crit is None else {"criteria": crit})
        decide_fork(m, "s", {"q": q})
    assert seen == [["No.", "Yes."]] * 3 + [["No.", "threatens to cancel"]]
    assert ForkEngine._noul_texts({"false": "stays"}) == ["stays", "Yes."]



def test_fork_checks_every_question_before_running_any():
    import types
    from sieve.fork import decide as decide_fork
    runs = []
    m = types.SimpleNamespace(list_options=False, strip=False, max_len=50, _ids=lambda s: s.split(),
                              _probs=lambda state, instr, texts: runs.append(instr) or [0.5] * len(texts))
    ab = {"a": None, "b": None}
    with pytest.raises(ValueError, match="'long'"):            # the second question is over-long: nothing runs
        decide_fork(m, "s", {"short": {"type": "choice", "instructions": "Pick", "criteria": ab},
                             "long": {"type": "choice", "instructions": "x " * 100, "criteria": ab}})
    assert runs == []
    for q in ({"type": "choice", "instructions": "Pick", "criteria": {"a": None}},       # the common input rules
              {"type": "choice", "instructions": "", "criteria": ab}):
        with pytest.raises(ValueError):
            decide_fork(m, "s", {"q": q})
    assert runs == []


def test_sieve_4b_server_applies_the_common_input_rules():
    import types
    from fastapi.testclient import TestClient
    from sieve.codes.postprocess import assemble
    from sieve.codes.server import build_app
    calls = []

    def evaluate(req):
        calls.append(req)
        answers = {k: assemble(q, [0.33334, 0.33333, 0.33333] if q.type == "choice" else [0.4, 0.6], None)
                   for k, q in req.questions.items()}
        return {"model": "sieve-4b", "answers": answers, "usage": {"input_tokens": 9, "output_tokens": 0}}, {"model_ms": 1.0}
    engine = types.SimpleNamespace(model_name="sieve-4b", evaluate=evaluate, count_tokens=lambda text: len(text.split()))
    client = TestClient(build_app(engine))
    post = lambda state, q: client.post("/v1/systemone", json={"state": state, "model": "sieve-4b", "questions": {"q": q}})
    ab = {"a": None, "b": None}
    for q in ({"type": "choice", "instructions": "Pick", "criteria": {"a": None}},          # one option
              {"type": "score", "instructions": "Urgency?", "criteria": ["low"]},          # one level
              {"type": "choice", "criteria": ab},                                          # no instructions
              {"type": "noul", "instructions": "Spam?", "criteria": {"yes": "is spam"}}):  # a key the schema drops
        r = post("s", q)
        assert r.status_code == 400 and r.json()["detail"]["error_type"] == "invalid_request"
    assert calls == []                                                     # refused before the model runs
    r = post("s", {"type": "maybe", "instructions": "Pick"})               # the schema's own 422, located in the body
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][0] == "body"
    r = client.post("/v1/systemone", json={"model": "sieve-4b", "questions": {"q": {"type": "noul", "instructions": "Yes?"}}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "state"]   # a missing state stays missing
    r = post(42, {"type": "noul", "instructions": 7})                      # bare numbers, read as their text
    assert r.status_code == 200 and calls[-1].state == "42" and calls[-1].questions["q"].instructions == "7"
    r = post("s", {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None, "c": None}})
    a = r.json()["answers"]["q"]                                           # rounded once, as every Sieve model
    assert r.status_code == 200 and a["probabilities"] == {"a": 0.3333, "b": 0.3333, "c": 0.3333} and "stats" not in a
    assert r.json()["usage"]["output_tokens"] > 0


def test_graph_runner_prefills_the_state_only_for_a_graph(monkeypatch):
    import types
    import sieve.graphs as graphs
    calls = []
    m = types.SimpleNamespace(prefill_from_record=lambda rec: calls.append("prefill") or {"cache": None, "n": 1},
                              logits=lambda rec, prefix=None, temperature=None: calls.append(("logits", prefix)) or [])
    runner = graphs.CudaGraphRunner.__new__(graphs.CudaGraphRunner)
    runner.m, runner.max_graph_tokens, runner.graphs = m, 1024, {}
    long_row = [{"ids": list(range(1300))}]                               # past the graph limit
    monkeypatch.setattr(graphs, "rows_of", lambda rec: (None, None, long_row))
    runner.logits({"n_state": 41})
    assert calls == [("logits", None)]                                     # no prefill the fallback would throw away


def test_eval_skips_questions_the_models_refuse():
    from sieve.evaluate import items
    rec = {"state": "s", "questions": {
        "one": {"type": "choice", "instructions": "Pick", "criteria": {"a": None}, "label": "a"},
        "blank": {"type": "choice", "instructions": " ", "criteria": {"a": None, "b": None}, "label": "a"},
        "eleven": {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)], "label": 1}}}
    rows, skipped = items([rec], tok=None, max_state=10, max_branch=10, truncate_state=False)
    assert rows == [] and sum(skipped.values()) == 3 and all(k.startswith("invalid question") for k in skipped)


def test_sieve_4b_decision_index_engine_full_precision(monkeypatch):
    import types
    import sieve.codes.decision_index_engine as die
    from sieve.codes.postprocess import assemble
    seen = []
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(base="b", revision="r"), temperatures={})
    monkeypatch.setattr(die, "load", lambda *a, **kw: seen.append(kw["decimals"]) or types.SimpleNamespace(model=model))
    die.CodesEngine(model="m"); die.CodesEngine(model="m", decimals="4")        # the kit passes options as text
    assert seen == [None, 4]
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None, "c": None}}
    p = [0.33333, 0.33334, 0.33333]
    full, rounded = assemble(q, p, None), assemble(q, p, 4)
    assert full["choice"] == "b" == max(full["probabilities"], key=full["probabilities"].get)
    assert max(rounded["probabilities"], key=rounded["probabilities"].get) != rounded["choice"]   # the old default


def test_pointer_engine_retries_after_releasing_the_failed_pass(monkeypatch):
    import sys
    import types
    import weakref
    import torch
    if "decision_index" not in sys.modules:                                  # the kit is optional: stub its two names
        kit = types.ModuleType("decision_index"); engines = types.ModuleType("decision_index.engines")
        engines.Engine, engines.Unsupported = object, Exception
        kit.engines = engines
        monkeypatch.setitem(sys.modules, "decision_index", kit)
        monkeypatch.setitem(sys.modules, "decision_index.engines", engines)
    from sieve.decision_index_engine import SieveEngine
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}
    refs, alive = [], []
    e = SieveEngine.__new__(SieveEngine)
    e.max_state = e.max_branch = 16384
    e.provenance, e.T = {"model": "stub"}, 1.0
    e.m = types.SimpleNamespace(encode=lambda *a, **kw: {"ids": [1, 2]}, question_temperatures=lambda *a: None)

    def logits(*args, one_at_a_time=False):
        if not one_at_a_time:
            failed = torch.zeros(100)                       # a tensor of the pass that runs out of memory
            refs.append(weakref.ref(failed))
            raise torch.cuda.OutOfMemoryError("injected")
        alive.append(refs[0]() is not None)
        return [torch.tensor([0.0, 1.0])] * 2
    e._logits = logits
    e("s", {"one": q, "two": q})
    assert alive == [False]                                  # freed before the one-at-a-time retry


def test_eval_skips_malformed_entries_and_keeps_the_rest():
    from sieve.evaluate import items

    class Tok:                                                  # one id per character; markers get their own ids
        def __call__(self, text, add_special_tokens=False):
            return type("E", (), {"input_ids": [ord(c) % 1000 + 10 for c in text]})()

        def convert_tokens_to_ids(self, t):
            return 1 + sum(map(ord, t)) % 9
    good = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}, "label": "a"}
    records = [{"state": "s", "questions": {"bad": None, "num": 5, "flag": True, "good": good}},
               {"state": "s", "questions": None}, ["not", "a", "record"], {"state": "s", "questions": {"g2": good}}]
    rows, skipped = items(records, Tok(), max_state=1000, max_branch=1000, truncate_state=False)
    assert [r["question"] for r in rows] == ["good", "g2"]
    assert skipped == {"invalid question: not an object": 3, "invalid record: not an object with a questions object": 2}


class _EvalTok:                                                  # one id per character; markers get their own ids
    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) % 1000 + 10 for c in text]})()

    def convert_tokens_to_ids(self, t):
        return 1 + sum(map(ord, t)) % 9


def _eval_rows(records):
    import numpy as np
    from sieve.evaluate import items
    rows, skipped = items(records, _EvalTok(), 8192, 8192, False)
    for r in rows:
        r["logits"] = np.array([1.0, 0.0])
    return rows, skipped


def test_eval_groups_do_not_depend_on_their_names():
    from sieve.evaluate import accuracy_ci
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}
    def ci(metas):
        recs = [{"_meta": m, "state": "s", "questions": {"q": {**q, "label": lab}}} for m, lab in zip(metas, "ab")]
        return accuracy_ci(_eval_rows(recs)[0])
    assert ci([{"group_id": 0}, {"group_id": 0}]) == ci([{"group_id": 1}, {"group_id": 1}])      # 0 is a group too
    assert ci([{}, {"group_id": "rec0"}]) == ci([{}, {"group_id": "other"}])                     # no name collision


def test_eval_metadata_cannot_break_a_run():
    import json
    from sieve.evaluate import rounded, set_temperatures, summary
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}, "label": "a"}
    rec = lambda meta=None, src=None: {"state": "s", "_meta": meta or {}, "questions": {"q": {**q, "src": src}}}
    rows, skipped = _eval_rows([rec(src="valid"), rec({"group_id": [1]}), rec({"id": {"x": 1}}), rec(src=[1]),
                                rec(src={"n": 1}), rec(src=0), rec(src="dataset-A")])
    assert [r["src"] for r in rows] == ["valid", "0", "dataset-A"]
    assert skipped == {"invalid record: _meta group_id or id is not text or a number": 2,
                       "invalid question: src is not text or a number": 2}
    set_temperatures(rows, 1.0)
    out = summary(rows, 1.0)                                     # mixed numeric and text sources sort as text
    assert list(out["by_source"]) == ["0", "dataset-A", "valid"]
    json.dumps(rounded(out), allow_nan=False)


def test_server_prefix_cache_makes_room_before_a_new_prefix():
    import weakref
    import torch
    from sieve.serve import Request, Server

    class Model:                                                # memory for `limit` prefixes, no more
        device = "cpu"

        def __init__(self, limit):
            self.refs, self.ids, self.limit = [], {}, limit

        def encode(self, state, questions):
            return {"ids": [self.ids.setdefault(state, len(self.ids))] * 64 + [100, 101, 102], "n_state": 64}

        def question_temperatures(self, *a):
            return None

        def whole_row(self, rec):
            return False

        def prefill_from_record(self, rec):
            if sum(r() is not None for r in self.refs) >= self.limit:
                raise torch.OutOfMemoryError("no room")
            buf = torch.zeros(64)
            self.refs.append(weakref.ref(buf))
            return {"n": 64, "buffer": buf}

        def logits(self, *a, **kw):
            return [torch.tensor([1.0, 0.0])]
    q = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": None, "b": None}}}
    for limit, kept in ((8, 8), (3, 3)):                        # the cache holds 8; memory may hold fewer
        srv = Server(Model(limit))
        for i in range(9):
            assert srv.decide(Request(state=f"state-{i}", questions=q))["answers"]["q"]["choice"] == "a"
        assert len(srv.cache) == kept


def test_pointer_full_rows_keep_no_attention_cache():
    import types
    import torch
    from sieve.model import PointerHead, SieveModel
    seen = []

    class LM(torch.nn.Module):
        def forward(self, input_ids, use_cache=None, **kw):
            seen.append(use_cache)
            return types.SimpleNamespace(last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], 8))
    m = SieveModel.__new__(SieveModel)
    torch.nn.Module.__init__(m)
    m.lm, m.device, m.pad_id, m.head_dtype, m.head = LM(), "cpu", 0, torch.float32, PointerHead(8, 4).eval()
    with torch.no_grad():
        m.row_logits([{"ids": list(range(1, 9)), "opts": [3, 6], "decide": 7}])
        m.RECOMPUTE_MIN_BRANCH = 1                                 # a long question: state and question in one row
        m.logits({"ids": list(range(1, 9)), "pos": list(range(8)), "n_state": 1, "opt_idx": [[3, 6]], "decide_idx": [7]})
    assert seen == [False, False]


def test_server_frees_cached_states_when_a_question_pass_runs_out_of_memory():
    import weakref
    import torch
    from sieve.serve import Request, Server

    class Model:                                                # room for 3 cached states plus one pass
        device = "cpu"

        def __init__(self):
            self.refs, self.ids, self.failed = [], {}, None

        def live(self):
            return sum(r() is not None for r in self.refs)

        def encode(self, state, questions):
            return {"ids": [self.ids.setdefault(state, len(self.ids))] * 64 + [100, 101, 102], "n_state": 64}

        def question_temperatures(self, *a):
            return None

        def whole_row(self, rec):
            return False

        def prefill_from_record(self, rec):
            buf = torch.zeros(64)
            self.refs.append(weakref.ref(buf))
            return {"n": 64, "buffer": buf}

        def logits(self, *a, **kw):
            assert self.failed is None or self.failed() is None    # the failed pass is gone before the retry
            if self.live() >= 4:
                failed = torch.zeros(64)
                self.failed = weakref.ref(failed)
                raise torch.OutOfMemoryError("no workspace")
            return [torch.tensor([1.0, 0.0])]
    srv = Server(Model())
    q = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": None, "b": None}}}
    for i in range(6):
        assert srv.decide(Request(state=f"state-{i}", questions=q))["answers"]["q"]["choice"] == "a"
    assert len(srv.cache) == 3 and tuple([5] * 64) in srv.cache  # the request's own state stays cached


def test_server_caches_no_state_a_whole_row_ignores_and_retries_once_uncached():
    import weakref
    import torch
    from sieve.serve import Request, Server

    class Model:                                                # any live cached state leaves no room for a pass
        device = "cpu"

        def __init__(self, whole):
            self.whole, self.refs, self.prefills = whole, [], 0

        def encode(self, state, questions):
            return {"ids": [7] * 64 + [100, 101, 102], "n_state": 64}

        def question_temperatures(self, *a):
            return None

        def whole_row(self, rec):
            return self.whole

        def prefill_from_record(self, rec):
            self.prefills += 1
            buf = torch.zeros(64)
            self.refs.append(weakref.ref(buf))
            return {"n": 64, "buffer": buf}

        def logits(self, rec, prefix=None, temperature=None):
            if any(r() is not None for r in self.refs):
                raise torch.OutOfMemoryError("the cached state takes the workspace")
            return [torch.tensor([1.0, 0.0])]
    q = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": None, "b": None}}}
    whole = Server(Model(whole=True))                           # one long question: its state is never cached
    assert whole.decide(Request(state="s", questions=q))["answers"]["q"]["choice"] == "a"
    assert whole.m.prefills == 0 and not whole.cache
    srv = Server(Model(whole=False))                            # only its own state is cached: dropped, run once uncached
    assert srv.decide(Request(state="s", questions=q))["answers"]["q"]["choice"] == "a"
    assert srv.m.prefills == 1 and not srv.cache


def test_pointer_logits_build_no_autograd_graph():
    import types
    import torch
    from sieve.model import PointerHead, SieveModel
    modes = []

    class LM(torch.nn.Module):                                  # trainable weights: a graph would be built if allowed
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(32, 8)

        def forward(self, input_ids, **kw):
            modes.append(torch.is_grad_enabled())
            return types.SimpleNamespace(last_hidden_state=self.emb(input_ids))
    m = SieveModel.__new__(SieveModel)
    torch.nn.Module.__init__(m)
    m.lm, m.device, m.pad_id, m.head_dtype, m.head = LM(), "cpu", 0, torch.float32, PointerHead(8, 4).eval()
    m.RECOMPUTE_MIN_BRANCH = 1                                     # the one-row path; no outer torch.no_grad here
    z = m.logits({"ids": [i % 32 for i in range(12)], "pos": list(range(12)), "n_state": 4, "opt_idx": [[5, 9]],
                  "decide_idx": [11]})[0]
    assert modes == [False] and not z.requires_grad

def test_letters_engine_limits_and_full_precision_scores():
    import types
    import torch
    from sieve.letters.engine import LettersEngine
    class Stub:
        temperature, temperatures = 1.0, None
        fmt = types.SimpleNamespace(label_ids=[1, 2, 3])                 # a format with 3 labels
        def logits_many(self, requests, budget):
            return [[torch.tensor([0.1, 0.7, 0.3])]]
    e = LettersEngine.__new__(LettersEngine)
    e.m, e.budget, e.provenance = Stub(), 1, {"model": "x"}
    with pytest.raises(ValueError, match="declared limit 3"):
        e("s", {"q": {"type": "choice", "instructions": "Pick", "criteria": {k: None for k in "abcd"}}})
    a = e("s", {"q": {"type": "score", "instructions": "Rate", "criteria": ["lo", "mid", "hi"]}})[0]["answers"]["q"]
    p = torch.softmax(torch.tensor([0.1, 0.7, 0.3]), -1).tolist()
    assert a["probabilities"]["0"] == p[0] and a["score"] == sum(i * x for i, x in enumerate(p))     # not rounded


def test_slots_answers_from_unrounded_rows(monkeypatch):
    import sys
    import types
    names = {"decider": {}, "decider.infer": {"Decider": object}, "decider.temperature": {},
             "decider.prompt": {"MAX_OPTIONS": 255, "chat_template": None, "letter_ids": None},
             "decider.systemone": {"assemble": None, "unique_tokens": None,
                                   "combine_isolated": lambda f: ([x / sum(f) for x in f], sum(f))}}
    for n, attrs in names.items():                                        # decider-ai is optional: stub what is used
        if n not in sys.modules:
            mod = types.ModuleType(n); mod.__dict__.update(attrs); monkeypatch.setitem(sys.modules, n, mod)
    monkeypatch.delitem(sys.modules, "sieve.slots.engine", raising=False)
    from sieve.slots.engine import SlotsEngine
    rows = [[0.62137, 0.27911, 0.09952] + [0.0] * 252,                     # team: a choice of 3, padded label slots
            [0.30881, 0.69119],                                            # leave: noul (no, yes)
            [0.6, 0.4], [0.2, 0.8], [0.7, 0.3]]                            # anger: one yes/no row per level ("iso")
    stats = {"index": [["team", "list", 0, 1], ["leave", "list", 1, 1], ["anger", "iso", 2, 3]], "row_probs": rows}
    qs = {"team": {"type": "choice", "instructions": "Team?", "criteria": {"a": None, "b": None, "c": None}},
          "leave": {"type": "noul", "instructions": "Leaving?"},
          "anger": {"type": "score", "instructions": "Anger?", "criteria": ["calm", "angry", "furious"]}}
    out = SlotsEngine.answers(None, "s", qs, stats)
    level = [0.4 / 1.5, 0.8 / 1.5, 0.3 / 1.5]
    _, _, meta = to_record("s", qs)
    ref = to_answers([[x / sum(rows[0][:3]) for x in rows[0][:3]], rows[1], level], meta)
    assert out == ref and out["anger"]["score"] == round(sum(i * x for i, x in enumerate(level)), 4)


def test_sieve_4b_rounds_once():
    from sieve.api import standard_answers
    from sieve.codes.postprocess import assemble
    q = {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}
    p = [0.12345000000049, 0.87654999999951]
    raw = assemble(q, p, decimals=None)                                    # no rounding of its own
    assert raw["probabilities"] == {"a": p[0], "b": p[1]}
    out = standard_answers({"q": raw}, {"q": q})["q"]                      # rounded once, as every model
    _, _, meta = to_record("s", {"q": q})
    assert out == to_answers([p], meta)["q"] and abs(sum(out["probabilities"].values()) - 1) < 1e-9


def test_server_reports_design():
    import types
    from sieve.serve import Server
    srv = Server.__new__(Server)
    srv.model_id, srv.base, srv.graphed, srv.size, srv.hits, srv.misses, srv.cache = "m", "b", None, 8, 0, 0, {}
    srv.m = types.SimpleNamespace(device="cpu", head=types.SimpleNamespace(temperature=1.0), temperatures=None)
    assert srv.info()["design"] == "pointer"
