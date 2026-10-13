"""Decision Index engine for Sieve-0.8B, Sieve-2B and Sieve-2B-v2 (or a local copy of one of them).

    python -m decision_index run --engine sieve.decision_index_engine:SieveEngine \
        --option model=sthanika-ai/Sieve-2B-v2 --out runs/Sieve-2B-v2

Sieve.choice(state, question, options) already returns what a "choice" answer needs (argmax + a
full probability dict over the options); this is mostly plumbing: build option texts from
`criteria`, treat `noul` as a 2-option Choice (the model has no native yes/no primitive), and enforce
the Decision Index "no truncation -- raise Unsupported instead" rule here, because the model's own
`_trim()` silently drops the front of an overlong state rather than erroring.

Every question of a request is scored with the model's multi-prefix fork in as few GPU calls as
memory allows (`batched`, the default), or one question at a time (`--option batched=false`).
"""
import os

import torch

try:
    from decision_index.engines.base import Engine, Unsupported, text
except ImportError:                                  # the kit is only needed to run the Decision Index
    from .._kit import Engine, Unsupported, text

from .model import Sieve, choice_prefix, listed_question


class ForkEngine(Engine):
    name = "sieve"

    def __init__(self, model="", sieve_dir="", repo_id="", device=None, dtype="bfloat16", schema=None, **options):
        super().__init__(**options)
        path = model or repo_id or sieve_dir
        assert path, "pass --option model=<Hub repo or local folder>"
        revision = None
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            sieve_dir = snapshot_download(path, revision=options.get("revision"),
                                          allow_patterns=["config.json", "*.safetensors"])
            revision = sieve_dir.rstrip("/").rsplit("/", 1)[-1]
        else:
            sieve_dir = path
        self.model = Sieve.from_pretrained(sieve_dir, device=device, dtype=getattr(torch, dtype))
        self.name = f"sieve-{path.rstrip('/').rsplit('/', 1)[-1]}"
        self.schema = schema
        frac = float(options.get("mem_fraction", 0))
        if frac and self.model.device != "cpu":
            # parallel shards share a GPU (with each other and other users' jobs): cap each so one
            # can't OOM its neighbours; a request that doesn't fit backs off inside this process
            torch.cuda.set_per_process_memory_fraction(frac)
        self.batched = str(options.get("batched", "true")).lower() != "false"
        self.max_rows = int(options.get("max_rows", 256))
        self.max_tokens = int(options.get("max_batch_tokens", 65536))
        self.provenance = {
            "kind": "sieve",
            "path": path,
            "revision": revision,
            "device": self.model.device,
            "dtype": dtype,
            "list_options": self.model.list_options,
            "built_in_temperature": self.model.T,
            "context_limit_tokens": self.model.max_len,
            "schema_calibration": schema,
            "policy": "Order-invariant prefix-fork Choice decision on a Qwen3.5 hybrid backbone "
                      "(Gated DeltaNet + attention); every option is scored as an isolated "
                      "continuation of one shared, cached [state][question] prefix, so shuffling "
                      "the options changes nothing. noul is answered as a 2-option Choice over its "
                      "own true/false descriptions when given, else plain 'Yes.'/'No.' text. No "
                      "option is filtered and the rendering is fixed for every benchmark.",
        }

    def _option_texts(self, keys, criteria):
        return [key if criteria.get(key) in (None, "") else text(criteria[key]) for key in keys]

    def _would_truncate(self, state_str, question, options):
        """Mirrors Sieve._trim's own overflow check exactly, so we refuse before it ever
        silently trims: over = len(prefix) + longest_option - max_len."""
        q = listed_question(question, options) if self.model.list_options else question
        pre_ids = self.model.tok(choice_prefix(state_str, q, self.model.strip), add_special_tokens=False).input_ids
        longest = max(len(self.model.tok(o, add_special_tokens=False).input_ids) for o in options)
        return len(pre_ids) + longest - self.model.max_len > 0

    def _prepare(self, state_str, questions):
        """-> [(qid, qtype, keys, texts, instr)], raising Unsupported exactly as the per-question
        path does (unknown type, duplicate option text, context overflow)."""
        prepared = []
        for k, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                texts = self._option_texts(keys, q["criteria"])
            elif q["type"] == "noul":
                keys = ["false", "true"]
                crit = q.get("criteria") or {}
                texts = self._option_texts(keys, crit) if crit else ["No.", "Yes."]
            else:
                raise Unsupported("Unsupported question type " + str(q["type"]))
            if len(set(texts)) != len(texts):
                raise Unsupported(f"question {k!r}: duplicate option text, cannot be scored distinctly")
            prepared.append((k, q["type"], keys, texts, text(q.get("instructions", ""))))
        return prepared

    def _batched_probs(self, state_str, prepared):
        """Every question of the request in as few GPU calls as possible, using Sieve's own
        multi-prefix fork (the path its training/eval code uses): one _prefix_pass over all
        questions' [state][question] prefixes (left-padded -- exact for DeltaNet per Sieve), then
        one _fork_pass over all their options, each row continuing its own question's prefix via
        row_src. Each question's prefix is identical to Sieve.choice()'s, so results match the
        per-question path up to padding/batching float noise. Questions are grouped under a row and
        token budget; a group that still OOMs is split in half and retried."""
        import torch.nn.functional as F
        m = self.model
        items = []  # (pre_ids, [opt_ids])
        for _, _, _, texts, instr in prepared:
            qtext = listed_question(instr, texts) if m.list_options else instr
            pre = m._ids(choice_prefix(state_str, qtext, m.strip))
            opts = [m._ids(o) for o in texts]
            if len(pre) + max(len(o) for o in opts) > m.max_len:
                raise Unsupported(f"state+question+option exceeds Sieve's {m.max_len}-token context limit")
            items.append((pre, opts))

        # Group by prefix length: a batched prefix pass costs B * max_len(prefix), so mixing a
        # 1,500-token prefix with 200-token ones wastes most of the pass on padding (measured: naive
        # batching was 0.59x the per-question speed for exactly this reason). Sorted order +
        # a padded-token budget keeps each group's padding small; cost is counted as padded tokens.
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        groups, cur = [], []
        for i in order:
            cand = cur + [i]
            pmax = max(len(items[j][0]) for j in cand)
            omax = max(len(o) for j in cand for o in items[j][1])
            nrows = sum(len(items[j][1]) for j in cand)
            padded = len(cand) * pmax + nrows * (pmax + omax)
            waste = len(cand) * pmax - sum(len(items[j][0]) for j in cand)
            if cur and (nrows > self.max_rows or padded > self.max_tokens or waste > 0.25 * len(cand) * pmax):
                groups.append(cur)
                cand = [i]
            cur = cand
        if cur:
            groups.append(cur)

        probs = [None] * len(items)

        def run_opts(i, opts):
            with torch.no_grad():
                cache, pre_att = m._prefix_pass([items[i][0]])
                h = m._fork_pass(cache, pre_att, [len(items[i][0])], opts, [0] * len(opts))
                return (m.head(h) / m.T).float()

        def run_group(idx):
            with torch.no_grad():
                cache, pre_att = m._prefix_pass([items[i][0] for i in idx])
                opt_rows, row_src = [], []
                for b, i in enumerate(idx):
                    opt_rows += items[i][1]
                    row_src += [b] * len(items[i][1])
                h = m._fork_pass(cache, pre_att, [len(items[i][0]) for i in idx], opt_rows, row_src)
                return (m.head(h) / m.T).float()

        # Every retry happens AFTER leaving the except block, and each attempt runs in its own helper
        # frame: while an except block is active, the traceback keeps the failed attempt's frames --
        # and their replicated cache tensors -- alive, so a retry started inside it inherits that
        # memory and fails again (seen: every POP909 / API-Bank request "OOM at a single option").
        def option_logits(i, opts):
            """Logits for a slice of ONE question's options. Options are isolated continuations of
            the same prefix, so scoring them in chunks is exact; _fork_pass mutates its cache, so
            each chunk gets its own prefix pass. Halves down to one option before giving up."""
            try:
                return run_opts(i, opts)
            except torch.OutOfMemoryError:
                pass
            torch.cuda.empty_cache()
            if len(opts) == 1:
                # transient resource failure, not a declared capacity limit: surface it as a
                # retryable error (the runner retries `error` rows on resume), never as
                # `unsupported`, which would count it wrong for good
                raise RuntimeError("transient CUDA OOM scoring a single option; retry on resume")
            mid = len(opts) // 2
            return torch.cat([option_logits(i, opts[:mid]), option_logits(i, opts[mid:])])

        def score(idx):
            try:
                logits = run_group(idx)
            except torch.OutOfMemoryError:
                logits = None
            if logits is None:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    i = idx[0]
                    probs[i] = F.softmax(option_logits(i, items[i][1]), dim=-1).cpu().numpy()
                    return
                score(idx[:len(idx) // 2])
                score(idx[len(idx) // 2:])
                return
            j = 0
            for i in idx:
                n = len(items[i][1])
                probs[i] = F.softmax(logits[j:j + n], dim=-1).cpu().numpy()
                j += n

        for g in groups:
            score(g)
        return probs

    def __call__(self, state, questions):
        state_str = text(state) if state not in ("", None, {}, []) else ""
        if self.batched:
            prepared = self._prepare(state_str, questions)
            answers = {}
            for (k, qtype, keys, texts, instr), p in zip(prepared, self._batched_probs(state_str, prepared)):
                probs_by_key = {key: float(x) for key, x in zip(keys, p)}
                if qtype == "choice":
                    answers[k] = {"type": "choice", "choice": keys[int(p.argmax())], "probabilities": probs_by_key}
                else:
                    answers[k] = {"type": "noul", "noul": probs_by_key["true"]}
            return {"model": self.name, "answers": answers}, None

        answers = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                texts = self._option_texts(keys, q["criteria"])
            elif q["type"] == "noul":
                keys = ["false", "true"]
                crit = q.get("criteria") or {}
                texts = self._option_texts(keys, crit) if crit else ["No.", "Yes."]
            else:
                raise Unsupported("Unsupported question type " + str(q["type"]))

            if len(set(texts)) != len(texts):
                raise Unsupported(f"question {k!r}: duplicate option text, cannot be scored distinctly")

            instr = text(q.get("instructions", ""))
            if self._would_truncate(state_str, instr, texts):
                raise Unsupported(f"question {k!r}: state+question+option exceeds Sieve's "
                                   f"{self.model.max_len}-token context limit")

            out = self.model.choice(state_str, instr, texts, schema=self.schema)
            # Sieve.choice's "decision"/"probs" are keyed by option TEXT, not our criteria key --
            # map back positionally (texts and keys share the same order by construction).
            probs_by_key = dict(zip(keys, out["probs"].values()))
            chosen_key = keys[out["index"]]
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": chosen_key, "probabilities": probs_by_key}
            else:
                answers[k] = {"type": "noul", "noul": probs_by_key["true"]}
        return {"model": self.name, "answers": answers}, None
