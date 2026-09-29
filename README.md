# Sieve

Sieve is a family of decision models. Give a model a piece of text or JSON (the *state*) and typed questions, and it
returns a calibrated probability for every option of every question. It reads the state once, scores the options
directly, and never generates text. Each model is a LoRA adapter and a small pointer head on an open Qwen backbone.

| model | backbone | weights | Decision Index 0.2.1 |
|---|---|---|--:|
| Sieve-9B | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [`sthanika-ai/Sieve-9B`](https://huggingface.co/sthanika-ai/Sieve-9B) | 41.71 ¹ |

More models will be added to the [Sieve collection](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586) on Hugging Face. All of them run on this code, which
reads each model's backbone from its `adapter_config.json`.

¹ Self-reported, not yet on the leaderboard (see [Decision Index](#decision-index)).

## How it works

Sieve builds on the decision-model design of [Kev](https://github.com/jaredpalmer/kev):

- **Backbone.** Only the text part of the Qwen model is kept, so the model cannot produce text.
  - **Sieve-9B keeps:** 32 layers and the embeddings, 7.94B parameters.
  - **Sieve-9B drops:** the language-model head (1.02B), the multi-token-prediction layer (0.24B) and the vision
    encoder (0.46B).
- **Readout.** The state, each question and its options form one sequence. A 2.1M-parameter pointer head scores the
  hidden state at the end of each option against the one at the decision point. A calibrated softmax gives the
  probabilities.
- **Isolation.** Each question runs as its own row from the state's cache, so questions never see each other.
  Options can change on every request.

## What Sieve changes

- **Sieve-9B:**
  - it uses the post-trained Qwen3.5-9B instead of the base model;
  - its rank-32 adapter and pointer head are the exact average of two rank-16 runs from the same initialisation;
  - it trained on 1,920 extra records from 240 rule structures not in the decision-v7 split, and on states padded with
    unrelated text.
- **Serving (all models):**
  - CUDA graphs remove the kernel-launch overhead of short requests (Sieve-9B, 2 options: 70.2 → 18.6 ms);
  - a single long question (1,024+ tokens) runs as one causal row on the fused attention kernel (255 options:
    371.0 → 330.5 ms).

## Install

```bash
pip install "sieve-decisions[cuda,serve] @ git+https://github.com/sthanika-ai/sieve"
# or the exact tested environment:
pip install -r requirements.txt && pip install -e .
```

Sieve-9B needs a CUDA GPU with about 20 GB of memory.

## Quickstart

```python
from sieve import load_sieve, decide

m = load_sieve("sthanika-ai/Sieve-9B")          # any Sieve model, or a local folder
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
# department: billing 0.8957 (confidence 0.8436) · urgency: 1.8037 of 0–2 · churn_risk: 0.975
```

| type | options | returns |
|---|---|---|
| `choice` | the caller's keys, optionally described (up to 255) | the most likely key, a confidence, a probability per key |
| `noul` | yes / no | P(yes) |
| `score` | an ordered list of levels | the expected level, a confidence, a probability per level |

Each model's `head.pt` stores a temperature fitted on held-out data. It never changes an answer; `temperature=1.0`
gives the raw distribution.

## HTTP server

```bash
sieve-serve --model sthanika-ai/Sieve-9B --graphs --port 8020
curl -s localhost:8020/v1/decide -H 'content-type: application/json' -d '{"state": "Billed twice. Refund today or we cancel.",
  "questions": {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "this week", "today"]}}}'
```

Repeated states of 64+ tokens come from a prefix cache (last 8 states), and `GET /v1/models` reports the model and
cache state.

Server-reported p50 for Sieve-9B on one A100 80GB, with one choice question and a cached state. Tokenisation and
HTTP are excluded.

| options | 2 | 5 | 10 | 25 | 50 | 100 | 255 |
|---|--:|--:|--:|--:|--:|--:|--:|
| ms | 18.6 | 19.3 | 22.8 | 42.8 | 74.5 | 128.4 | 330.5 |

## Evaluate

```bash
sieve-eval --model sthanika-ai/Sieve-9B --data benchmark.jsonl --out result.json
```

- **Input format.** Each line of `benchmark.jsonl` is a request (`state`, `questions`) plus a gold `label` for every
  question: a choice key, `true`/`false`, or a score level index. The files in `data/` use this format.
- **Metrics.**
  - accuracy, with a 95% bootstrap interval, and the error rate;
  - NLL, Brier and ECE (15 bins);
  - the share of questions answered wrongly with p ≥ 0.9.

  These are reported at the stored temperature and raw, with accuracy broken down by question type and by each
  question's `src`.
- **Limits.** Questions over the token limits are skipped and counted, not truncated, unless you pass
  `--truncate-state`.
- **Predictions.** `--predictions preds.jsonl` also writes every question's probabilities.
- **Reproducing our numbers.** With `--no-merge`, `sieve-eval` reproduces Sieve-9B's public-test numbers in the
  model card exactly. We checked SciQ, Banking77 and typed-decisions: every answer and every metric is identical.

## Decision Index

Sieve-9B scores **41.71** on our own run of the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
0.2.1, using the official [kit](https://github.com/apolinario/decision-index):

- **Coverage.** The run has 150,317 requests over 44 benchmarks, all answered; the index averages 38 of those
  benchmarks.
- **Not confirmed.** The score is not on the leaderboard yet, so it is expected, not confirmed. On the 2026-09-28
  board it would place about 18th of 72.
- **No truncation.** Nothing is truncated. The declared limits are a 65,536-token state, a 32,768-token question and
  255 options.

```bash
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=sthanika-ai/Sieve-9B --out runs/Sieve-9B
python -m decision_index score --results runs/Sieve-9B/results.jsonl
```

## Training data

[`data/sieve-9b/`](data/sieve-9b/) holds Sieve-9B's four training files, unchanged:

| file | records | contents |
|---|--:|---|
| `decision_v7_train.jsonl` | 12,576 | 1,000 records from each of ten public datasets, 896 policy minimal pairs, 1,680 rule-structure records |
| `date_policy.jsonl` | 900 | date-policy cases |
| `date_policy_unknowable.jsonl` | 1,425 | the same 900 cases, 256 with the deciding evidence removed, and 269 controls |
| `rule_structures.jsonl` | 1,920 | 240 rule structures, 8 records each (30 are logically equivalent to a decision-v7 one) |

Each record has three fields:

- `state`;
- `questions`, with a gold `label` or a `target` distribution;
- `_meta`, which says where the record came from.

The recipe and hashes are in the model's `training_config.json`, and the sources and licences are in
[NOTICE](NOTICE).

## Credits and license

- **Kev** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev), Apache-2.0):
  - the decision format, the pointer head and the stored temperature;
  - the input delimiters, escaping and confidence formulas in the code;
  - Sieve-9B's training data and rule generator.
- **Others:**
  - ten public datasets, listed in [NOTICE](NOTICE);
  - the Qwen3.5-9B backbone (Apache-2.0);
  - "Model soups" (Wortsman et al., 2022).
- **License:** the code is Apache-2.0 ([LICENSE](LICENSE)). The files in `data/` keep their sources' licences.

Tests: `pip install -e ".[test]" && pytest`.
