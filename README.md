# Sieve — decision models

Sieve is a family of decision models. Give a model a piece of text or JSON (the *state*) and typed questions, and it returns a calibrated probability for every option of every question. It reads the state once, scores the options directly, and never generates text.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-56BF4F?style=flat-square&labelColor=1E281F)](LICENSE)
[![Sieve-27B](https://img.shields.io/badge/%F0%9F%A4%97%20model-Sieve--27B-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/sthanika-ai/Sieve-27B)
[![Sieve-9B-Plus](https://img.shields.io/badge/%F0%9F%A4%97%20model-Sieve--9B--Plus-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/sthanika-ai/Sieve-9B-Plus)
[![Sieve-9B](https://img.shields.io/badge/%F0%9F%A4%97%20model-Sieve--9B-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/sthanika-ai/Sieve-9B)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20collection-Sieve-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586)

## What Sieve is for

Many steps in a product are decisions, not writing. Examples are routing a ticket, classifying a document, checking a request against a policy, or picking the tool to call. Sieve answers these directly:

- **Answers that always parse.** The caller supplies the options, and every option gets a probability, so there is no free text to parse or validate.
- **Calibrated probabilities.** Each model ships a temperature fitted on held-out data, so a probability can be used as a threshold. Newer models fit one temperature per question category.
- **Fast.** The state is read once and every option is scored in the same pass. Repeated states come from a cache.
- **Flexible.** Options can change on every request, up to 255 per question. Several questions can share one state without seeing each other.

## How it works

- **Backbone.** Each model is a LoRA adapter on an open Qwen backbone. Only the text part of the backbone is used, so the model cannot produce text.
- **Readout.** The state, each question and its options form one sequence. A small pointer head scores the hidden state at the end of each option against the one at the decision point, and a calibrated softmax gives the probabilities.
- **Isolation.** Each question runs as its own row from the state's cache, so questions never see each other.

## Models

| model | backbone | GPU memory | Decision Index 0.3 ¹ | Decision Index 0.2.1 ¹ | weights and model card |
|---|---|--:|--:|--:|---|
| Sieve-27B | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | ~55 GB | [**57.06**](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-decision-index-results) | 57.21 | [`sthanika-ai/Sieve-27B`](https://huggingface.co/sthanika-ai/Sieve-27B) |
| Sieve-9B-Plus | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | ~20 GB | [**53.35**](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-Plus-decision-index-results) | – | [`sthanika-ai/Sieve-9B-Plus`](https://huggingface.co/sthanika-ai/Sieve-9B-Plus) |
| Sieve-9B | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | ~20 GB | – | [41.71](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-decision-index-results) | [`sthanika-ai/Sieve-9B`](https://huggingface.co/sthanika-ai/Sieve-9B) |

¹ Our own runs of the public [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) suite with the official [kit](https://github.com/apolinario/decision-index), self-reported and not yet on the leaderboard. Sieve-9B-Plus was run on 0.3 only, and Sieve-9B has not been run on 0.3 yet.

Results, latency and training details are on each model's card. All models run on this code. It reads each model's backbone from the model's `adapter_config.json` and loads that backbone at a pinned revision. New models are added to the [Sieve collection](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586).

## Install

```bash
pip install "sieve-decisions[cuda,serve] @ git+https://github.com/sthanika-ai/sieve"
# or the exact tested environment:
pip install -r requirements.txt && pip install -e .
```

Recommended: `pip install --no-build-isolation causal-conv1d==1.7.0`, the fused convolution kernel for the backbones' linear-attention layers. Without it, transformers uses a slower PyTorch fallback, whose probabilities can differ slightly.

## Use

```python
from sieve import load_sieve, decide

m = load_sieve("sthanika-ai/Sieve-27B")          # any Sieve model, or a local folder
decide(m,
    state={"subject": "Duplicate charge on invoice 4411",
           "body": "Billed twice for March. Refund today or we cancel the annual contract."},
    questions={
        "department": {"type": "choice", "instructions": "Which team handles this?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages",
                                    "sales": "pricing and contracts"}},
        "urgency":    {"type": "score", "instructions": "How urgent is this?",
                       "criteria": ["can wait", "this week", "today"]},
        "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
    })
```

| type | options | returns |
|---|---|---|
| `choice` | the caller's keys, optionally described (up to 255) | the most likely key, a confidence, a probability per key |
| `noul` | yes / no | P(yes) |
| `score` | an ordered list of levels | the expected level, a confidence, a probability per level |

- **Temperature.** The model's `head.pt` stores the temperature. It never changes an answer, and `temperature=1.0` gives the raw distribution.
- **Temperature per category.** Sieve-9B-Plus also stores one temperature per question category. A category is the question's type, its number of options (2, 3-4, 5-9, 10-25, 26+) and the kind of state (empty, text or JSON), so it is known from the request alone. A question takes the temperature of its most specific category in the table (`choice:3-4:text`, then `choice:3-4`, then `choice`), else the global one. Models without the table, such as Sieve-27B and Sieve-9B, use the global temperature as before. Passing `temperature=` overrides them all.
- **Example script.** [`examples/quickstart.py`](examples/quickstart.py) runs the request above.

## HTTP server

```bash
sieve-serve --model sthanika-ai/Sieve-27B --graphs --port 8020
curl -s localhost:8020/v1/decide -H 'content-type: application/json' -d '{"state": "Billed twice. Refund today or we cancel.",
  "questions": {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "this week", "today"]}}}'
```

- **Cache.** Repeated states of 64 or more tokens come from a prefix cache, which holds the last 8 states.
- **Status.** `GET /v1/models` reports the model, its temperatures and the cache state.
- **CUDA graphs.** `--graphs` replays short requests as CUDA graphs, which removes kernel-launch overhead.

## Evaluate

```bash
sieve-eval --model sthanika-ai/Sieve-27B --data benchmark.jsonl --out result.json
```

- **Input format.** Each line of `benchmark.jsonl` is a request (`state`, `questions`) plus a gold `label` for every question: a choice key, `true`/`false`, or a score level index.
- **Metrics.**
  - accuracy, with a 95% bootstrap interval, and the error rate;
  - NLL, Brier and ECE (15 bins);
  - the share of questions answered wrongly with p ≥ 0.9.

  They are reported at the stored temperature and raw, with accuracy broken down by question type and by each question's `src`. For a model with per-category temperatures, they are also reported at the global temperature and per category.
- **Limits.** Questions over the token limits are skipped and counted, not truncated, unless you pass `--truncate-state`.
- **Predictions.** `--predictions preds.jsonl` also writes every question's probabilities.

### Decision Index

Any Sieve model runs through the official [Decision Index kit](https://github.com/apolinario/decision-index) with the engine in this package:

```bash
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=sthanika-ai/Sieve-27B --out runs/Sieve-27B
python -m decision_index score --results runs/Sieve-27B/results.jsonl
```

The engine runs the model as it is served. Nothing is truncated and no option is dropped. Requests over the declared limits (a 65,536-token state, a 32,768-token question, 255 options) are reported as unsupported.

## Training data

[`data/`](data/) holds Sieve-9B's training files; [`data/README.md`](data/README.md) lists what each file contains, where it comes from and its licence. The training data of the other models is not published.

## Tests

```bash
pip install -e ".[test]" && pytest
```

## Citation

```bibtex
@misc{sieve2026,
  title  = {Sieve: Decision Models},
  author = {{Sthanika AI}},
  year   = {2026},
  url    = {https://github.com/sthanika-ai/sieve}
}
```

## License

Apache-2.0 for the code, see [LICENSE](LICENSE). The files in `data/` keep their sources' licences, see [`data/README.md`](data/README.md).
