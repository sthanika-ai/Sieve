# Training data

## `sieve-9b/`

Sieve-9B's four training files, unchanged:

| file | records | contents |
|---|--:|---|
| `decision_v7_train.jsonl` | 12,576 | 1,000 records from each of ten public datasets, 896 policy minimal pairs, 1,680 rule-structure records |
| `date_policy.jsonl` | 900 | date-policy cases |
| `date_policy_unknowable.jsonl` | 1,425 | the same 900 cases, 256 with the deciding evidence removed, and 269 controls |
| `rule_structures.jsonl` | 1,920 | 240 rule structures, 8 records each |

Each record has three fields: `state`; `questions`, with a gold `label` or a `target` distribution; and `_meta`, which says where the record came from.

## Sources and licences

The files come from Kev (https://github.com/jaredpalmer/kev, Apache-2.0):

- `decision_v7_train.jsonl` is Kev's decision-v7 training split, from the Hugging Face dataset `jaredpalmer/kev-suites`, whose card states no licence.
- The two `date_policy` files are from Kev's `evals/night2/`.
- `rule_structures.jsonl` was generated with Kev's rule generator (`kev/composition.py`).

`decision_v7_train.jsonl` contains text from ten public datasets, which keep the licences stated on their Hugging Face cards. Each record's `_meta` names its source, revision and row.

| licence | datasets |
|---|---|
| cc-by-4.0 | Banking77 (`legacy-datasets/banking77`) |
| cc-by-sa-3.0 | BoolQ (`google/boolq`), DBpedia-14 (`fancyzhx/dbpedia_14`) |
| mixed | MultiNLI (`nyu-mll/multi_nli`: cc-by-3.0, cc-by-sa-3.0, mit, other) |
| apache-2.0 | Amazon Reviews en (`SetFit/amazon_reviews_multi_en`) |
| other | IMDB (`stanfordnlp/imdb`), Yelp Review Full (`Yelp/yelp_review_full`) |
| unknown | AG News (`fancyzhx/ag_news`), TREC (`CogComp/trec`) |
| not stated | SST-5 (`SetFit/sst5`) |

Check these licences before any commercial use of the data.
