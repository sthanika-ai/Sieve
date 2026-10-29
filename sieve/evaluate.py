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

from .api import question_keys, to_record, validate_questions
from .calibration import categories
from .encode import TooLong, encode, rows_of

BINS = 15


def gold_index(q):
    """The gold option's index; ValueError (the question is skipped) for a label that is not one of the question's:
    choice, one of its keys; noul, true / false (also "true" / "false", 1 / 0); score, a level index 0..L-1 (an int,
    or a whole number written as 2.0 or "2")."""
    keys = question_keys(q["type"], q.get("criteria"))
    label = q["label"]
    if q["type"] == "choice":
        return keys.index(label)
    if q["type"] == "noul":
        for i, names in enumerate(((False, "false"), (True, "true"))):
            if label in names and not (isinstance(label, (int, float)) and not isinstance(label, bool) and label != i):
                return i
        raise ValueError(f"noul label {label!r} is not true or false")
    i = None                                        # a level index: an int, or a whole number written as 2.0 or "2"
    if isinstance(label, int) and not isinstance(label, bool):
        i = label
    elif isinstance(label, float) and label.is_integer():
        i = int(label)
    elif isinstance(label, str) and label.strip().isdigit():
        i = int(label)
    if i is None or not 0 <= i < len(keys):
        raise ValueError(f"score label {label!r} is not a level index 0..{len(keys) - 1}")
    return i


def probs(z, t):
    z = np.asarray(z, float) / t
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def metrics(rows, t):
    """t: one temperature for every row, or None for each row's own r["t"] (per-category calibration)."""
    P = [probs(r["logits"], r["t"] if t is None else t) for r in rows]
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


def _scalar(v):
    return v is None or isinstance(v, (str, int, float))


def items(records, tok, max_state, max_branch, truncate_state):
    out, skipped = [], defaultdict(int)
    for i, r in enumerate(records):
        if not isinstance(r, dict) or not isinstance(r.get("questions"), dict):
            skipped["invalid record: not an object with a questions object"] += 1
            continue
        meta = r.get("_meta") if isinstance(r.get("_meta"), dict) else {}
        if not (_scalar(meta.get("group_id")) and _scalar(meta.get("id"))):
            skipped["invalid record: _meta group_id or id is not text or a number"] += 1
            continue
        # the bootstrap's group: a supplied group_id, else the record's id (0 counts), else the record alone, in a
        # namespace of its own so that it never joins a supplied group of the same name
        if meta.get("group_id") is not None:
            group = ("supplied", meta["group_id"])
        elif meta.get("id") is not None:
            group = ("supplied", meta["id"])
        else:
            group = ("record", i)
        for qid, q in r["questions"].items():
            if not isinstance(q, dict):
                skipped["invalid question: not an object"] += 1
                continue
            if "label" not in q:
                skipped["no label"] += 1
                continue
            if not _scalar(q.get("src")):
                skipped["invalid question: src is not text or a number"] += 1
                continue
            try:
                validate_questions({qid: q})            # the input rules the models are served by
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
            out.append({"record": meta.get("id", i), "question": qid, "group": group,
                        "src": None if q.get("src") is None else str(q["src"]),
                        "type": q["type"], "label": label, "k": len(row["opts"]),
                        "cats": categories(q["type"], len(row["opts"]), r["state"]),
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


def set_temperatures(rows, t, table=None):
    """Each row's temperature: its finest category in the per-category table, else the global t."""
    for r in rows:
        r["t"] = next((table[c] for c in r.get("cats", ()) if c in table), t) if table else t
    return rows


def summary(rows, t, per_category=False):
    """per_category: rows carry their own r["t"]; the global t is then reported beside it."""
    cal, raw = metrics(rows, None if per_category else t), metrics(rows, 1.0)
    out = OrderedDict([("questions", len(rows)), ("accuracy", cal["accuracy"]), ("accuracy_ci95", accuracy_ci(rows)),
                       ("error_rate", 1 - cal["accuracy"]),
                       ("calibrated", {k: v for k, v in cal.items() if k != "accuracy"}),
                       ("raw", {k: v for k, v in raw.items() if k != "accuracy"})])
    if per_category:
        out["calibrated_global_temperature"] = {k: v for k, v in metrics(rows, t).items() if k != "accuracy"}
        parts = defaultdict(list)
        for r in rows:
            parts[r["cats"][0]].append(r)
        out["by_category"] = OrderedDict((k, {"questions": len(v), "temperature": v[0]["t"], **metrics(v, None)})
                                         for k, v in sorted(parts.items()))
    for key, name in (("type", "by_type"), ("src", "by_source")):
        parts = defaultdict(list)
        for r in rows:
            if r[key] is not None:
                parts[r[key]].append(r)
        if parts:
            out[name] = OrderedDict((k, {"questions": len(v), "accuracy": metrics(v, None if per_category else t)["accuracy"]})
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
    ap.add_argument("--max-state", type=int, default=None, help="state token limit (default: the model's trained limit)")
    ap.add_argument("--max-question", type=int, default=None,
                    help="question, options and delimiters (default: the model's trained limit)")
    ap.add_argument("--truncate-state", action="store_true", help="cut states over --max-state instead of skipping them")
    ap.add_argument("--batch-tokens", type=int, default=8192)
    a = ap.parse_args()

    from .load import family_of, load_pointer
    if family_of(a.model) != "pointer":
        raise SystemExit("sieve-eval works with the pointer-head models (Sieve-27B-v2, Sieve-27B, Sieve-9B-v2, Sieve-9B, Sieve-2B-v3); "
                         "use the Decision Index engine or decide() for the others")
    m = load_pointer(a.model, base=a.base, merge=not a.no_merge)
    t = float(m.head.temperature)
    with open(a.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    max_state, max_question = a.max_state or m.max_state, a.max_question or m.max_branch
    rows, skipped = items(records, m.tok, max_state, max_question, a.truncate_state)
    if not rows:
        raise SystemExit("no scoreable questions in the data file")
    score(m, rows, a.batch_tokens)
    table = (m.temperatures or {}).get("table")
    set_temperatures(rows, t, table)
    result = OrderedDict([("model", a.model), ("data", a.data), ("records", len(records)), ("skipped", skipped),
                          ("temperature", t), ("merged", not a.no_merge),
                          ("max_state", max_state), ("max_question", max_question), ("truncate_state", a.truncate_state)])
    if table:
        result["temperatures"] = m.temperatures
    result.update(summary(rows, t, per_category=bool(table)))
    with open(a.out, "w") as f:
        json.dump(rounded(result), f, indent=1, ensure_ascii=False)
    if a.predictions:
        with open(a.predictions, "w") as f:
            for r in rows:
                line = {"record": r["record"], "question": r["question"], "label": r["label"],
                        "probabilities": [round(float(x), 6) for x in probs(r["logits"], r["t"])]}
                if table:                                   # per-category calibration: the temperature this question used
                    line["temperature"] = round(float(r["t"]), 6)
                f.write(json.dumps(line) + "\n")
    print(json.dumps({k: result[k] for k in ("questions", "accuracy", "accuracy_ci95")}), "->", a.out)


if __name__ == "__main__":
    main()
