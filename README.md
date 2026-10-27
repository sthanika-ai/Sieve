# Sieve — decision models

Sieve is a family of decision models. Give a model a piece of text or JSON (the *state*) and typed questions, and it returns a calibrated probability for every option of every question. It reads the state once, scores the options directly, and never generates text.

This repository holds the code for every Sieve model. The weights, model cards and Decision Index results are on Hugging Face, in the [Sieve collection](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586).

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-56BF4F?style=flat-square&labelColor=1E281F)](LICENSE)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20collection-Sieve-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/collections/sthanika-ai/sieve-6abb93a61dd495f69811c586)
[![Decision Index](https://img.shields.io/badge/leaderboard-Decision%20Index-FFD21E?style=flat-square&labelColor=1E281F)](https://huggingface.co/spaces/multimodalart/jev-decision-index)

## What Sieve is for

Many steps in a product are decisions, not writing. Examples are routing a ticket, classifying a document, checking a request against a policy, or picking the tool to call. Sieve answers these directly:

- **Answers that always parse.** The caller supplies the options, and every option gets a probability, so there is no free text to parse or validate.
- **Calibrated probabilities.** Each model ships temperatures fitted on held-out data, so a probability can be used as a threshold.
- **Fast.** The state is read once and every option is scored in the same pass.
- **Flexible.** Options can change on every request, up to 255 per question. Several questions can share one state without seeing each other.

## Models

| model | backbone | design | Full score ¹ | Public score ¹ | weights and model card |
|---|---|---|--:|--:|---|
| Sieve-31B-v3 | [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | letter readout | – ² | [71.68](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-v3-decision-index-results) ² | [`sthanika-ai/Sieve-31B-v3`](https://huggingface.co/sthanika-ai/Sieve-31B-v3) |
| Sieve-31B-v2 | [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | answer slots | – ² | [69.97](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-v2-decision-index-results) ² | [`sthanika-ai/Sieve-31B-v2`](https://huggingface.co/sthanika-ai/Sieve-31B-v2) |
| Sieve-31B | [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | answer slots | **64.25** | [62.52](https://huggingface.co/datasets/sthanika-ai/Sieve-31B-decision-index-results) | [`sthanika-ai/Sieve-31B`](https://huggingface.co/sthanika-ai/Sieve-31B) |
| Sieve-27B-v2 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | pointer head | **59.89** | [68.02](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-v2-decision-index-results) | [`sthanika-ai/Sieve-27B-v2`](https://huggingface.co/sthanika-ai/Sieve-27B-v2) |
| Sieve-27B | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | pointer head | 60.17 ³ | [60.04](https://huggingface.co/datasets/sthanika-ai/Sieve-27B-decision-index-results) | [`sthanika-ai/Sieve-27B`](https://huggingface.co/sthanika-ai/Sieve-27B) |
| Sieve-9B-v2 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | pointer head | **47.00** | [53.35](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-v2-decision-index-results) | [`sthanika-ai/Sieve-9B-v2`](https://huggingface.co/sthanika-ai/Sieve-9B-v2) |
| Sieve-9B | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | pointer head | **41.73** | [41.88](https://huggingface.co/datasets/sthanika-ai/Sieve-9B-decision-index-results) | [`sthanika-ai/Sieve-9B`](https://huggingface.co/sthanika-ai/Sieve-9B) |
| Sieve-4B | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | option codes | **40.96** | [50.06](https://huggingface.co/datasets/sthanika-ai/Sieve-4B-decision-index-results) | [`sthanika-ai/Sieve-4B`](https://huggingface.co/sthanika-ai/Sieve-4B) |
| Sieve-2B-v3 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | pointer head | **34.45** | [53.51](https://huggingface.co/datasets/sthanika-ai/Sieve-2B-v3-decision-index-results) | [`sthanika-ai/Sieve-2B-v3`](https://huggingface.co/sthanika-ai/Sieve-2B-v3) |
| Sieve-2B-v2 | [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | option fork | 30.84 ³ | [37.26](https://huggingface.co/datasets/sthanika-ai/Sieve-2B-v2-decision-index-results) | [`sthanika-ai/Sieve-2B-v2`](https://huggingface.co/sthanika-ai/Sieve-2B-v2) |
| Sieve-2B | [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) | option fork | **21.90** | [22.16](https://huggingface.co/datasets/sthanika-ai/Sieve-2B-decision-index-results) | [`sthanika-ai/Sieve-2B`](https://huggingface.co/sthanika-ai/Sieve-2B) |
| Sieve-0.8B | [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | option fork | **20.82** | [32.26](https://huggingface.co/datasets/sthanika-ai/Sieve-0.8B-decision-index-results)  | [`sthanika-ai/Sieve-0.8B`](https://huggingface.co/sthanika-ai/Sieve-0.8B) |

¹ From the [Decision Index leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index), edition 0.4.1 (board data of 2026-10-26). The Full score ranks the board: it comes from the maintainers' private tests, 60% tests of the same skills as the public benchmarks and 40% decision tasks from new domains. The Public score is the public suite alone; each one links to the model's complete results.

² Not on the leaderboard yet: our own run of the public 0.3 suite with the official [kit](https://github.com/apolinario/decision-index).

³ No longer on the leaderboard, where its successor replaced it (Sieve-27B-v2 replaced Sieve-27B, Sieve-2B-v3 replaced Sieve-2B-v2). Full score from edition 0.3.2 (board data of 2026-10-14), which counted the public benchmarks 20%, the private tests of the same skills 50% and the private tasks from new domains 30%.

Results, latency and training details are on each model's card.

### Sieve-Vision

The Sieve-Vision models answer multiple-choice questions about one or more images, with a calibrated probability for every option, and never write text. Their inference code is in their model repositories (`sieve_vision/`), not in this package; models and results are in the [Sieve-Vision collection](https://huggingface.co/collections/sthanika-ai/sieve-vision-6ad61203c0d53c2a58cc2f5b).

| model | backbone | Full score ⁴ | Public score ⁴ | weights and model card |
|---|---|--:|--:|---|
| Sieve-Vision-27B | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | – ⁵ | [73.12](https://huggingface.co/datasets/sthanika-ai/Sieve-Vision-27B-decision-index-results) ⁵ | [`sthanika-ai/Sieve-Vision-27B`](https://huggingface.co/sthanika-ai/Sieve-Vision-27B) (CC BY-NC 4.0) |
| Sieve-Vision-4B | [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | **58.60** | [65.17](https://huggingface.co/datasets/sthanika-ai/Sieve-Vision-4B-decision-index-results) | [`sthanika-ai/Sieve-Vision-4B`](https://huggingface.co/sthanika-ai/Sieve-Vision-4B) |

⁴ From the Decision Index vision leaderboard, edition 0.4 (board data of 2026-10-26): the Full score is half the public vision suite and half the maintainers' private vision tests.

⁵ Not on the leaderboard yet: our own run of the public vision suite.

## How it works

Every model reads a state and typed questions and returns a probability for every option. Only the text part of the backbone runs, so no model can produce text. The models come in five designs:

- **Pointer head** (Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3). A LoRA adapter on a Qwen backbone (Sieve-27B-v2 and Sieve-2B-v3: fully fine-tuned; their trained layers are copied into the backbone in place when loaded). The state, each question and its options form one sequence. A small pointer head scores the hidden state at the end of each option against the one at the decision point. Each question runs as its own row from the state's cache, so questions never see each other.
- **Option fork** (Sieve-2B-v2, Sieve-2B, Sieve-0.8B). The first 16 blocks of a Qwen3.5 base, adapted with LoRA (Sieve-0.8B: fully fine-tuned). The state and the question are encoded once, the encoding is copied once per option, and each option is scored as its own continuation by a scalar head. No option sees another, so reordering the options cannot change the answer.
- **Option codes** (Sieve-4B). A LoRA adapter on Qwen3.5-4B. The options are listed under codes (A, B, …, AA, …), and a 255-way head, initialised from the backbone's own output rows for those codes, reads the decision point. A yes/no question is asked with its options in both orders and the two answers are averaged.
- **Letter readout** (Sieve-31B-v3). Gemma-4-31B-it, fully fine-tuned and stored as a full model with a `head.pt`. All the questions of a request share one prompt in the model's chat template: a shared block (system turn, the list of questions, the state), then one block per question with its labelled options and an `Answer: (` slot, where the model's own output-layer logits of the option labels give the answer. In the 10 global attention layers the shared block attends only to itself and each question's block attends to the shared block and to itself, in both directions, so questions never see each other; the 50 sliding-window layers stay causal, as in training. A `choice` question whose options may be reordered is read twice, with its options as given and rotated by half, and the two log-probability vectors are averaged.
- **Answer slots** (Sieve-31B-v2, Sieve-31B). Gemma-4-31B-it with a LoRA adapter (Sieve-31B) or fully fine-tuned (Sieve-31B-v2: its decoder layers are copied into the backbone in place when loaded, and it uses bidirectional attention in its 10 global attention layers: they read the whole prompt in both directions, so the state is read in the light of the question and options that follow it, while the 50 sliding-window layers stay causal, as in training). Each question goes in the model's chat template with thinking off, followed by an answer slot, and the output rows of the option labels score that slot.

`load_sieve` tells the designs apart from a model's files, so the same calls work for every model.

## Install

```bash
pip install "sieve-decisions[cuda,serve] @ git+https://github.com/sthanika-ai/Sieve"
# or the exact tested environment:
pip install -r requirements.txt && pip install -e .
```

- **Fused kernels.** Recommended for the Qwen3.5 and Qwen3.8 backbones: `pip install --no-build-isolation causal-conv1d==1.7.0`, the fused convolution kernel for their linear-attention layers. Without it, transformers uses a slower PyTorch fallback, whose probabilities can differ slightly.
- **Sieve-31B-v2 and Sieve-31B** also need their prompt builder, installed without its dependencies: `pip install "sieve-decisions[slots] @ git+https://github.com/sthanika-ai/Sieve" && pip install --no-deps decider-ai==1.6.0`.
- **GPU memory** in bf16: 58 GiB of weights for Sieve-31B-v3 (its longest public Decision Index request, 53,703 tokens, peaked at 81 GiB: one 96 GB GPU), about 83 GB for Sieve-31B and 84 GB for Sieve-31B-v2 (one 96 GB GPU, or two GPUs with `split`), 52 GB for Sieve-27B-v2, 55 GB for Sieve-27B, 20 GB for Sieve-9B-v2 and Sieve-9B, 10 GB for Sieve-4B, and 5 to 11 GB for Sieve-2B-v3. The other 2B models and Sieve-0.8B need a few GB.

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

- **Question types.** Every model answers `choice` and `noul`. The option-fork models (Sieve-2B-v2, Sieve-2B, Sieve-0.8B) do not answer `score`. Sieve-31B-v3, Sieve-27B-v2 and Sieve-2B-v3 answer `score`, but their training data had only `choice` and `noul` questions. Some models add a few fields to an answer beyond those above.
- **Temperatures.** Each model stores its calibrated temperatures with its weights. They never change an answer. Sieve-27B and Sieve-9B have one temperature in `head.pt`. Sieve-31B-v3 (whose `head.pt` holds them), Sieve-27B-v2, Sieve-9B-v2 and Sieve-2B-v3 also have one per question category: the question's type, its number of options (2, 3-4, 5-9, 10-25, 26+) and the kind of state (empty, text or JSON). A question takes the temperature of its most specific category in the table (`choice:3-4:text`, then `choice:3-4`, then `choice`), else the global one. Sieve-4B has one per question type, the option-fork models have one in `config.json`, and Sieve-31B-v2 and Sieve-31B use 8.5495, their training temperature. `load_sieve(..., temperature=1.0)` gives the raw distribution. For the pointer-head models, `decide(..., temperature=...)` also overrides it for one request.
- **Token limits.** A request over a model's limits is refused, never truncated:
  - pointer head: the limits each model was trained with, from its `training_config.json`. That is 16,384 tokens for the state and 16,384 for each question with its options for Sieve-27B-v2, Sieve-27B, Sieve-9B-v2 and Sieve-2B-v3, and 8,192 for Sieve-9B. `m.encode(..., max_state=...)` and `sieve-eval --max-state` override them.
  - option fork: 32,768 tokens for the state, the question and the longest option.
  - Sieve-4B: 32,768 tokens for the state plus the longest question and 65,536 per request, with up to 10 score levels.
  - Sieve-31B-v2 and Sieve-31B: 32,768 tokens for each question's prompt.
  - Sieve-31B-v3: none declared (it was trained on prompts of up to 16,384 tokens); a request that does not fit in GPU memory fails with an error.
- **Example script.** [`examples/quickstart.py`](examples/quickstart.py) runs the request above on Sieve-9B, which needs about 20 GB of GPU memory.

## HTTP server

```bash
sieve-serve --model sthanika-ai/Sieve-27B --graphs --port 8020
curl -s localhost:8020/v1/decide -H 'content-type: application/json' -d '{"state": "Billed twice. Refund today or we cancel.",
  "questions": {"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "this week", "today"]}}}'
```

- **Any model.** `--model` takes any Sieve model. The pointer-head models get the prefix cache and CUDA graphs below; the others are answered through `decide`.
- **Cache.** Repeated states of 64 or more tokens come from a prefix cache, which holds the last 8 states.
- **Status.** `GET /v1/models` reports the model, its temperatures and the cache state.
- **CUDA graphs.** `--graphs` replays short requests as CUDA graphs, which removes kernel-launch overhead.
- **Sieve-4B's own server.** `python -m sieve.codes.server --ckpt sthanika-ai/Sieve-4B --port 8123` serves Sieve-4B on a `/v1/systemone` endpoint, with a state cache of its own.

## Evaluate

```bash
sieve-eval --model sthanika-ai/Sieve-27B --data benchmark.jsonl --out result.json
```

`sieve-eval` works with the pointer-head models (Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3).

- **Input format.** Each line of `benchmark.jsonl` is a request (`state`, `questions`) plus a gold `label` for every question: a choice key, `true`/`false`, or a score level index.
- **Metrics.**
  - accuracy, with a 95% bootstrap interval, and the error rate;
  - NLL, Brier and ECE (15 bins);
  - the share of questions answered wrongly with p ≥ 0.9.

  They are reported at the stored temperature and raw, with accuracy broken down by question type and by each question's `src`. For a model with per-category temperatures, they are also reported at the global temperature and per category.
- **Limits.** The model's trained token limits are the defaults. Questions over them are skipped and counted, not truncated, unless you pass `--truncate-state`.
- **Predictions.** `--predictions preds.jsonl` also writes every question's probabilities.

### Decision Index

Every Sieve model runs through the official [Decision Index kit](https://github.com/apolinario/decision-index) with one engine from this package:

```bash
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=sthanika-ai/Sieve-27B --out runs/Sieve-27B
python -m decision_index score --results runs/Sieve-27B/results.jsonl
```

The engine reads the model's design from its files and runs the model as it is served. Nothing is truncated and no option is dropped; a request over a model's declared limits is reported as unsupported.

- **Pointer head.** The adapter is merged into the bf16 backbone (a full fine-tune's layers are copied in place). The declared limits are a 65,536-token state, a 32,768-token question and 255 options; in our runs of Sieve-27B and Sieve-9B-v2, 241 requests had a question longer than 16,384 tokens and were answered in full.
- **Option fork.** bf16, with every question of a request scored in as few passes as memory allows. `--option mem_fraction=0.16` caps one process's share of a GPU when several share it.
- **Sieve-4B.** Each request is answered as its `/v1/systemone` server answers it: bf16, adapter merged, yes/no asked in both orders.
- **Sieve-31B-v3.** bf16, served by `sieve.letters` as described above: the prompt, the attention rule, the two option orders and the per-category temperatures come from `head.pt`. In a request with several questions and a shared block of 2,048 tokens or more, the shared block is computed once and its keys and values serve every question and both orders, which gives the same result as computing it with each question. No token limit is declared; up to 255 options.
- **Sieve-31B-v2 and Sieve-31B.** The backbone loads at its pinned revision. Sieve-31B's adapter stays unmerged; Sieve-31B-v2's layers are copied in place and its global layers run with bidirectional attention (read from `"attention": "bidir_global"` in its `training_config.json`). `--option split='{"0": "40GiB", "1": "40GiB"}'` spreads the model over two GPUs.

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
  url    = {https://github.com/sthanika-ai/Sieve}
}
```

## License

Apache-2.0, see [LICENSE](LICENSE).
