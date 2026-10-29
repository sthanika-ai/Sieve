# Evaluation

## sieve-eval

```bash
sieve-eval --model sthanika-ai/Sieve-27B-v2 --data benchmark.jsonl --out result.json
```

`sieve-eval` works with the pointer-head models (Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3).

- **Input format.** Each line of `benchmark.jsonl` is a request (`state`, `questions`) plus a gold `label` for every question: a choice key, `true`/`false`, or a score level index.
- **Metrics.**
  - accuracy, with a 95% bootstrap interval, and the error rate;
  - NLL, Brier and ECE (15 bins);
  - the share of questions answered wrongly with p ≥ 0.9.

  They are reported at the stored temperature and raw, with accuracy broken down by question type and by each question's `src`. For a model with per-category temperatures, they are also reported at the global temperature and per category.
- **Limits.** The model's trained token limits are the defaults. Questions over them are skipped and counted, not truncated, unless you pass `--truncate-state`.
- **Input rules.** A question that breaks the input rules every model is served by (`sieve.api.validate_questions`), such as a one-option choice or empty instructions, is skipped and counted with its reason, as is a record or question that is not a JSON object, a `_meta` `group_id` or `id`, or a question's `src`, that is not text or a number. Sources are reported as text.
- **Predictions.** `--predictions preds.jsonl` also writes every question's probabilities.

## Decision Index

Every Sieve model runs through the official [Decision Index kit](https://github.com/apolinario/decision-index) with one engine from this package. Install the kit and build its suite once, as its README describes (the suite is not redistributed), then:

```bash
python -m decision_index run --engine sieve.decision_index_engine:SieveEngine --option model=sthanika-ai/Sieve-27B-v2 --out runs/Sieve-27B-v2
python -m decision_index score --results runs/Sieve-27B-v2/results.jsonl
```

The engine reads the model's design from its files and runs the model as it is served. Nothing is truncated and no option is dropped; a request over a model's declared limits is reported as unsupported.

- **Pointer head.** The adapter is merged into the bf16 backbone (a full fine-tune's layers are copied in place). The declared limits are a 65,536-token state, a 32,768-token question and 255 options; in our runs of Sieve-27B and Sieve-9B-v2, 241 requests had a question longer than 16,384 tokens and were answered in full.
- **Option fork.** bf16, with every question of a request scored in as few passes as memory allows. `--option mem_fraction=0.16` caps one process's share of a GPU when several share it.
- **Sieve-4B.** Each request is computed as its `/v1/systemone` server computes it: bf16, adapter merged, yes/no asked in both orders. Probabilities are at full precision. The published run went through the server, which then rounded them to 4 decimals summing to one with the remainder on one option, so in 14 of its 320,687 choice answers the chosen option is not the one with the highest reported probability; `--option decimals=4` gives that rounding.
- **Sieve-31B-v3.** bf16, served by `sieve.letters` as described above: the prompt, the attention rule, the two option orders and the per-category temperatures come from `head.pt`. In a request with several questions and a shared block of 2,048 tokens or more, the shared block is computed once and its keys and values serve every question and both orders, which gives the same result as computing it with each question. No token limit is declared; up to 255 options.
- **Sieve-31B-v2 and Sieve-31B.** The backbone loads at its pinned revision. Sieve-31B's adapter stays unmerged; Sieve-31B-v2's layers are copied in place and its global layers run with bidirectional attention (read from `"attention": "bidir_global"` in its `training_config.json`). `--option split='{"0": "40GiB", "1": "40GiB"}'` spreads the model over two GPUs.
