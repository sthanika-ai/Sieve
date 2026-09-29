"""Evaluate a Sieve model on labelled typed-decision records and write the metrics as JSON.

    sieve-eval --model sthanika-ai/Sieve-9B --data benchmark.jsonl --out result.json

Each line of the data file is {"state": ..., "questions": {id: {type, instructions, criteria, label}}}, the request
format plus a gold "label" per question (a choice key, true/false, or a score level index). An optional "_meta"
object may carry "group_id" (related records share a group in the bootstrap) and "id".
"""
import argparse
import json
from collections import OrderedDict, defaultdict

import numpy as np
import torch

from .api import question_keys, to_record
from .encode import TooLong, encode, rows_of

BINS = 15


def gold_index(q):
    keys = question_keys(q["type"], q.get("criteria"))
    if q["type"] == "choice":
        return keys.index(q["label"])
    if q["type"] == "noul":
        return 1 if q["label"] in (True, "true", 1) else 0
    return int(q["label"])


def probs(z, t):
    z = np.asarray(z, float) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def metrics(rows, t):
    P = [probs(r["logits"], t) for r in rows]
    y = [r["label"] for r in rows]
    conf = np.array([p.max() for p in P])
    corr = np.array([int(np.argmax(p) == l) for p, l in zip(P, y)])
    nll = float(np.mean([-np.log(max(p[l], 1e-12)) for p, l in zip(P, y)]))
    brier = float(np.mean([((p - np.eye(len(p))[l]) ** 2).sum() for p, l in zip(P, y)]))
    ece, edges = 0.0, np.linspace(0, 1, BINS + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            ece += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return {"accuracy": float(corr.mean()), "nll": nll, "brier": brier, "ece": float(ece),
            "wrong_at_p_ge_0.9": float(np.mean((conf >= 0.9) & (corr == 0)))}


def accuracy_ci(rows, n=4000, seed=0):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(int(np.argmax(r["logits"]) == r["label"]))
    g = list(groups.values())
    s = np.array([sum(x) for x in g])
    c = np.array([len(x) for x in g])
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    d = s[idx].sum(1) / c[idx].sum(1)
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def items(records, tok, max_state, max_branch, truncate_state):
    out, skipped = [], defaultdict(int)
    for i, r in enumerate(records):
        meta = r.get("_meta") or {}
        group = meta.get("group_id") or meta.get("id") or f"rec{i}"
        for qid, q in r["questions"].items():
            if "label" not in q:
                skipped["no label"] += 1
                continue
            try:
                text, qs, _ = to_record(r["state"], {qid: q})
                label = gold_index(q)
                rec = encode(tok, text, qs, max_state=max_state, max_branch=max_branch, truncate_state=truncate_state)
            except TooLong:
                skipped["over the token limits"] += 1
                continue
            except (ValueError, KeyError) as e:
                skipped[f"invalid question: {str(e).split(':')[-1].strip()[:60]}"] += 1
                continue
            state_ids, _, (row,) = rows_of(rec)
            Ls = len(state_ids)
            out.append({"record": meta.get("id", i), "question": qid, "group": group, "src": q.get("src"),
                        "type": q["type"], "label": label, "k": len(row["opts"]),
                        "ids": state_ids + row["ids"], "opts": [Ls + o for o in row["opts"]], "decide": Ls + row["decide"]})
    return out, dict(skipped)


@torch.no_grad()
def score(m, rows, batch_tokens=8192):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    batches, cur = [], []
    for i in order:
        trial = cur + [i]
        if cur and len(trial) * len(rows[i]["ids"]) > batch_tokens:
            batches.append(cur)
            cur = [i]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    for b in batches:
        z, _ = m.row_logits([rows[i] for i in b])
        for j, i in enumerate(b):
            rows[i]["logits"] = z[j, :rows[i]["k"]].float().cpu().numpy()
    return rows


def summary(rows, t):
    cal, raw = metrics(rows, t), metrics(rows, 1.0)
    out = OrderedDict([("questions", len(rows)), ("accuracy", cal["accuracy"]), ("accuracy_ci95", accuracy_ci(rows)),
                       ("error_rate", 1 - cal["accuracy"]),
                       ("calibrated", {k: v for k, v in cal.items() if k != "accuracy"}),
                       ("raw", {k: v for k, v in raw.items() if k != "accuracy"})])
    for key, name in (("type", "by_type"), ("src", "by_source")):
        parts = defaultdict(list)
        for r in rows:
            if r[key] is not None:
                parts[r[key]].append(r)
        if parts:
            out[name] = OrderedDict((k, {"questions": len(v), "accuracy": metrics(v, t)["accuracy"]})
                                    for k, v in sorted(parts.items()))
    return out


def rounded(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return type(x)((k, rounded(v, nd)) for k, v in x.items())
    if isinstance(x, list):
        return [rounded(v, nd) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Sieve model on a labelled JSONL file.")
    ap.add_argument("--model", required=True, help="local folder or Hugging Face repo id")
    ap.add_argument("--data", required=True, help="JSONL file of labelled records")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--predictions", help="also write one JSON line per question with its probabilities")
    ap.add_argument("--base", default=None, help="backbone (default: the one in the model's adapter_config.json)")
    ap.add_argument("--no-merge", action="store_true", help="keep the adapter unmerged (exact, but slower)")
    ap.add_argument("--max-state", type=int, default=8192)
    ap.add_argument("--max-question", type=int, default=8192, help="question, options and delimiters")
    ap.add_argument("--truncate-state", action="store_true", help="cut states over --max-state instead of skipping them")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    a = ap.parse_args()

    from .load import load_sieve
    m = load_sieve(a.model, base=a.base, merge=not a.no_merge)
    t = float(m.head.temperature)
    with open(a.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    rows, skipped = items(records, m.tok, a.max_state, a.max_question, a.truncate_state)
    if not rows:
        raise SystemExit("no scoreable questions in the data file")
    score(m, rows, a.batch_tokens)
    result = OrderedDict([("model", a.model), ("data", a.data), ("records", len(records)), ("skipped", skipped),
                          ("temperature", t), ("merged", not a.no_merge),
                          ("max_state", a.max_state), ("truncate_state", a.truncate_state)])
    result.update(summary(rows, t))
    with open(a.out, "w") as f:
        json.dump(rounded(result), f, indent=1, ensure_ascii=False)
    if a.predictions:
        with open(a.predictions, "w") as f:
            for r in rows:
                f.write(json.dumps({"record": r["record"], "question": r["question"], "label": r["label"],
                                    "probabilities": [round(float(x), 6) for x in probs(r["logits"], t)]}) + "\n")
    print(json.dumps({k: result[k] for k in ("questions", "accuracy", "accuracy_ci95")}), "->", a.out)


if __name__ == "__main__":
    main()
