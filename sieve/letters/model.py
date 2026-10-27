"""Serving engine of a letter-readout model (Sieve-31B-v3): the model's own LM-head logits of the option labels at
"Answer: (" (final logit softcap included), over sieve.letters' renderer, the one the trainer uses.

Attention rule A, in the global layers of a bidirectional model (sliding-window layers stay causal in every case):
  - a request with one question is one block from <bos> to the answer slot: its state reads its question and options;
  - with several questions, block 0 (system turn, the list of questions, the state) is followed by one block per
    question: a question's block reads block 0 and itself, block 0 reads only itself, no question reads another. The
    questions are packed after one copy of block 0 (positions restart after it), exactly as the trainer's packs.
Single or multi follows the request's own question count. The same code serves a causal model (attention "causal",
e.g. the untuned base, the proxy reference), where the packs compute what each question's own row would.

Loads a release folder (save_pretrained + head.pt with readout "letters"), a trainer weight snapshot (snapshot.json +
backbone shards, on top of its base) or a base model; averages and blends of snapshots are built in memory (mix).
Requests are never truncated: one longer than max_tokens is reported as unsupported.

Serving tests (never trained): reads=2 reads a request of at most reread_max tokens twice (sieve.letters.twice, each
question one block); orders=2 averages the log-probabilities of the served option order and the order rotated by n // 2
(choice questions whose options may be permuted)."""
import glob
import json
import os

import numpy as np
import torch

from ..api import to_record
from ..calibration import question_temperatures
from .attention import mark_attention, seg_masks
from .format import LetterFormat
from .layout import escape_of


class Unsupported(Exception):
    """A request longer than the engine's validated limit (never truncated)."""


# The decoder MLPs of a long serving pass run in slices of MLP_CHUNK tokens: an MLP acts on each token alone, so this is
# the same computation, and a 64k-token pass (32 questions with long instructions) no longer needs several ~3 GB
# intermediates at once (it ran out of memory on an 80 GB A100). Serving only: the trainer's passes stay <= 16,384.
MLP_CHUNK = int(os.environ.get("SIEVE_MLP_CHUNK", "8192"))


def chunk_mlps(lm, chunk=MLP_CHUNK):
    """Wrap every decoder layer's MLP so that, without gradients, a sequence longer than chunk runs in slices."""
    for layer in getattr(lm, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "sieve_chunked", False):
            continue
        fwd = mlp.forward

        def sliced(x, *args, _fwd=fwd, **kw):
            if torch.is_grad_enabled() or x.dim() != 3 or x.shape[1] <= chunk:
                return _fwd(x, *args, **kw)
            return torch.cat([_fwd(x[:, s:s + chunk], *args, **kw) for s in range(0, x.shape[1], chunk)], 1)
        mlp.forward = sliced
        mlp.sieve_chunked = True


def _backbone(full):
    lm = full.model
    for attr in ("language_model", "text_model"):
        if hasattr(lm, attr):
            return getattr(lm, attr)
    return lm


def _snapshot_tensors(path):
    """{name: (file, path)} of a trainer weight snapshot."""
    idx = json.load(open(os.path.join(path, "backbone.safetensors.index.json")))["weight_map"]
    return {k: os.path.join(path, f) for k, f in idx.items()}


def _base_of(snap, base):
    if base:
        return base
    info = json.load(open(os.path.join(snap, "snapshot.json")))
    if os.path.isdir(info.get("base_path", "")):
        return info["base_path"]
    from huggingface_hub import snapshot_download
    return snapshot_download(info["base"], revision=info["base_revision"])


@torch.no_grad()
def mix_into(lm, sources):
    """lm's backbone weights <- sum_i w_i * source_i, sources [(snapshot folder or "base", w)] with the w summing to 1
    ("base" = lm's weights as loaded). Every tensor is mixed in fp32 and stored in lm's dtype."""
    from safetensors import safe_open
    assert abs(sum(w for _, w in sources) - 1.0) < 1e-6, "mix weights must sum to 1"
    maps = [(None if s == "base" else _snapshot_tensors(s), w) for s, w in sources]
    sd = lm.state_dict()
    files = {}
    for k, t in sd.items():
        if not t.is_floating_point():
            continue
        acc = None
        for mp, w in maps:
            if mp is None:
                x = t.float()
            else:
                f = mp[k]
                if f not in files:
                    files[f] = safe_open(f, "pt", device="cpu")
                x = files[f].get_tensor(k).float()
            acc = x.to(t.device) * w if acc is None else acc.add_(x.to(t.device), alpha=w)
        t.copy_(acc.to(t.dtype))
    missing = [k for mp, _ in maps if mp for k in sd if k not in mp]
    assert not missing, f"snapshot lacks {missing[:5]}"


class LetterModel:
    # a serving pass holds block 0 once and as many question blocks as fit PACK_TOKENS (exact under rule A at any size):
    # requests whose block 0 alone is long (32 questions with long instructions in the header list, ~30k tokens) would
    # otherwise recompute block 0 for every question; the trainer's packs stay at 16,384 (its prompts are capped there)
    PACK_TOKENS = int(os.environ.get("SIEVE_LETTERS_PACK_TOKENS", "49152"))   # a 64k pass did not fit an 80 GB A100
    # A request with several questions and a state of at least PREFIX_MIN tokens runs block 0 once (state_once): one pass
    # over block 0 keeps every layer's keys and values, and the branches of both option orders run after it, reading
    # them, in passes of at most BRANCH_TOKENS branch tokens. Rule A makes this the packs' attention exactly (block 0
    # never reads a branch, a branch reads block 0 and itself), without block 0 again in every pack and every order.
    PREFIX_MIN = int(os.environ.get("SIEVE_LETTERS_PREFIX_MIN", "2048"))
    BRANCH_TOKENS = int(os.environ.get("SIEVE_LETTERS_BRANCH_TOKENS", "32768"))

    def __init__(self, full, tok, attention, meta=None, device="cuda", max_tokens=None, pack_tokens=None):
        self.full, self.tok, self.device = full, tok, device
        self.lm = _backbone(full)
        self.lm.eval()
        cfg = getattr(full.config, "text_config", full.config)
        self.fmt = LetterFormat(tok)
        self.esc = escape_of(tok)
        self.windowed = "sliding_attention" in set(getattr(cfg, "layer_types", None) or [])
        self.window = cfg.sliding_window if self.windowed else None
        if self.windowed:
            mark_attention(self.lm)
        chunk_mlps(self.lm)
        if attention not in ("bidirectional", "causal"):
            raise ValueError(f"attention must be 'bidirectional' or 'causal', got {attention!r}")
        if attention == "bidirectional" and not self.windowed:
            raise ValueError("bidirectional global attention is implemented for sliding-window backbones (Gemma 4) only")
        self.bidir = attention == "bidirectional"
        self.softcap = float(getattr(cfg, "final_logit_softcapping", 0.0) or 0.0)
        W = full.get_output_embeddings().weight
        hd = torch.float64 if W.dtype == torch.float64 else torch.float32      # label logits in fp32 (fp64 for tests)
        self.W = W[self.fmt.label_ids].detach().to(hd)        # the LM head rows of the labels (tied embeddings)
        self.meta = meta or {}
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.temperatures = self.meta.get("temperatures")
        self.max_tokens, self.pack_tokens = max_tokens, pack_tokens or self.PACK_TOKENS
        self.policy = {"reads": 1, "reread_max": 3000, "orders": 1, **(self.meta.get("serving") or {})}
        self.pad_id = tok.pad_token_id or 0
        fmt_saved = self.meta.get("format")
        if fmt_saved:                                           # the renderer must be the one the model was trained with
            mine = self.fmt.describe()
            bad = [k for k in ("system", "header", "tail", "label_ids") if k in fmt_saved and fmt_saved[k] != mine[k]]
            if bad:
                raise ValueError(f"the model was trained with another letter format ({bad}); sieve.letters.format differs")

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path=None, *, base=None, mix=None, attention=None, device="cuda", dtype=torch.bfloat16, **kw):
        """path: a release folder (head.pt readout "letters"), a trainer snapshot (snapshot.json) or a base model.
        mix: [(snapshot folder or "base", weight)] built on top of base (or the first snapshot's base); attention:
        required for a base model, else read from head.pt / snapshot.json (an explicit value must agree)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        meta, snaps = {}, []
        if mix:
            snaps = [s for s, _ in mix if s != "base"]
            src = _base_of(snaps[0], base) if snaps else base
            info = json.load(open(os.path.join(snaps[0], "snapshot.json"))) if snaps else {}
            stored = info.get("global_attention")
        elif path and os.path.exists(os.path.join(path, "snapshot.json")):
            snaps, mix = [path], [(path, 1.0)]
            src = _base_of(path, base)
            stored = json.load(open(os.path.join(path, "snapshot.json"))).get("global_attention")
        elif path and os.path.exists(os.path.join(path, "head.pt")):
            src = path
            meta = torch.load(os.path.join(path, "head.pt"), map_location="cpu", weights_only=False)
            if meta.get("readout") != "letters":
                raise ValueError(f"{path}/head.pt is not a letter-readout model (use sieve.load_sieve)")
            stored = meta.get("global_attention", "causal")
        else:
            src, stored = path or base, None
        for s in snaps:                                         # every mixed snapshot must share the attention rule
            g = json.load(open(os.path.join(s, "snapshot.json"))).get("global_attention")
            if stored and g != stored:
                raise ValueError(f"snapshots mix attention rules ({g} vs {stored})")
        if attention and stored and attention != stored:
            raise ValueError(f"the model was trained with {stored} global attention, not {attention}")
        attention = attention or stored
        if attention is None:
            raise ValueError("a base model needs attention='causal' or 'bidirectional'")
        tok = AutoTokenizer.from_pretrained(src)
        # a release (no mix) loads straight onto its GPU: several engines on one machine never hold a CPU copy each
        direct = {"device_map": {"": device}} if not mix and str(device).startswith("cuda") else {}
        full = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, attn_implementation="sdpa", **direct)
        if mix:
            mix_into(_backbone(full), mix)
        full.to(device)
        meta.setdefault("global_attention", attention)
        return cls(full, tok, attention, meta, device=device, **kw)

    # ------------------------------------------------------------------ requests -> units
    def encode(self, state, questions):
        """(state ids, [instruction ids], [[option ids]], meta) of a request, rendered and escaped like the trainer."""
        text, qs, meta = to_record(state, questions)
        enc = lambda s: self.tok(self.esc(s), add_special_tokens=False).input_ids
        return enc(text), [enc(q["instr"]) for q in qs], [[enc(o) for o in q["options"]] for q in qs], meta

    def units(self, st, ins, opts, reads=1, reread_max=3000):
        """The passes of one request: [(ids, pos, seg, [(question, position, K)])], rule A."""
        fmt, Q = self.fmt, len(ins)
        block0 = fmt.state(ins, st)
        branches = [fmt.branch(ins[q], opts[q]) for q in range(Q)]
        if reads == 2 and len(block0) + sum(len(b) for b in branches) <= reread_max:
            out = []
            for q in range(Q):                                  # each question read twice is one block
                ids = fmt.twice(ins, st, ins[q], opts[q])
                out.append((ids, list(range(len(ids))), [0] * len(ids), [(q, len(ids) - 1, len(opts[q]))]))
            return out
        if Q == 1:
            ids = block0 + branches[0]
            return [(ids, list(range(len(ids))), [0] * len(ids), [(0, len(ids) - 1, len(opts[0]))])]
        ns = len(block0)
        lim = max(self.pack_tokens, ns + max(len(b) for b in branches))
        out, cur = [], None
        for q, b in enumerate(branches):                        # the trainer's packs: block 0 once, then the branches
            if cur is None or len(cur[0]) + len(b) > lim:
                cur = (list(block0), list(range(ns)), [0] * ns, [])
                out.append(cur)
            k = len(cur[3]) + 1
            cur[3].append((q, len(cur[0]) + len(b) - 1, len(opts[q])))
            cur[0].extend(b); cur[1].extend(range(ns, ns + len(b))); cur[2].extend([k] * len(b))
        return out

    # ------------------------------------------------------------------ passes
    def label_logits(self, h, K):
        z = h.to(self.W.dtype) @ self.W[:K].T
        return self.softcap * torch.tanh(z / self.softcap) if self.softcap else z

    @torch.no_grad()
    def run_units(self, units):
        """Raw label logits of every readout of a batch of units: [(question, logits)] per unit."""
        B, L = len(units), max(len(u[0]) for u in units)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        pos = torch.zeros((B, L), dtype=torch.long)
        seg = torch.full((B, L), -1, dtype=torch.long)
        for b, (i, p, s, _) in enumerate(units):
            n = len(i)
            ids[b, :n] = torch.tensor(i); pos[b, :n] = torch.tensor(p); seg[b, :n] = torch.tensor(s)
        ids, pos, seg = ids.to(self.device), pos.to(self.device), seg.to(self.device)
        h = self.lm(input_ids=ids, position_ids=pos, attention_mask=seg_masks(seg, pos, self.window, self.bidir),
                    use_cache=False).last_hidden_state
        return [[(q, self.label_logits(h[b, p], K).cpu()) for q, p, K in u[3]] for b, u in enumerate(units)]

    def check_length(self, units):
        n = max(len(u[0]) for u in units)
        if self.max_tokens and n > self.max_tokens:
            raise Unsupported(f"a pass of {n:,} tokens is over the validated limit {self.max_tokens:,} (not truncated)")

    def uses_state_once(self, n_state, n_questions, reads, short):
        """True for the requests state_once serves: several questions, block 0 of at least PREFIX_MIN tokens, a
        sliding-window backbone (its attention layers are marked), and not read twice."""
        return (self.windowed and n_questions > 1 and n_state >= self.PREFIX_MIN and self.PREFIX_MIN > 0
                and not (reads == 2 and short))

    @torch.no_grad()
    def state_once(self, block0, branch_sets):
        """Raw label logits of one request's branches, run after a single pass over its block 0.

        branch_sets: [(key, [(question, branch ids, K)])], e.g. the served option order and the rotated one. Returns
        {key: {question: logits}}. The block 0 pass keeps each marked layer's keys and values (sdpa_sieve); a sliding
        layer keeps only the last window - 1, all a branch can see. The branches then run concatenated, each with its
        own segment and positions restarting after block 0, reading those keys first, as in a pack."""
        dev, ns = self.device, len(block0)
        ks = min(ns, self.window - 1)
        marked = [m for m in self.lm.modules() if getattr(m, "sieve_attention", False)]
        cap = {"kv": {}, "sliding_keys": ks}
        pos = torch.arange(ns, device=dev)[None]
        for m in marked:
            m.sieve_capture = cap
        try:
            self.lm(input_ids=torch.tensor([block0], device=dev), position_ids=pos,
                    attention_mask=seg_masks(torch.zeros_like(pos), pos, self.window, self.bidir), use_cache=False)
        finally:
            for m in marked:
                m.sieve_capture = None
        out = {k: {} for k, _ in branch_sets}
        items = [(k, q, b, K) for k, bs in branch_sets for q, b, K in bs]
        try:
            for m in marked:
                m.sieve_prefix = cap["kv"][m]
            i = 0
            while i < len(items):
                j, n = i, 0
                while j < len(items) and (j == i or n + len(items[j][2]) <= self.BRANCH_TOKENS):
                    n += len(items[j][2]); j += 1
                ids, qpos, qseg, reads = [], [], [], []
                for s, (k, q, b, K) in enumerate(items[i:j], 1):
                    reads.append((k, q, len(ids) + len(b) - 1, K))
                    ids += b; qpos += range(ns, ns + len(b)); qseg += [s] * len(b)
                qp, qs = torch.tensor([qpos], device=dev), torch.tensor([qseg], device=dev)
                st = pos[:, ns - ks:]
                masks = seg_masks(qs, qp, self.window, self.bidir,
                                  k_seg=torch.cat([torch.zeros_like(pos), qs], 1), k_pos=torch.cat([pos, qp], 1),
                                  k_seg_sliding=torch.cat([torch.zeros_like(st), qs], 1), k_pos_sliding=torch.cat([st, qp], 1))
                h = self.lm(input_ids=torch.tensor([ids], device=dev), position_ids=qp, attention_mask=masks,
                            use_cache=False).last_hidden_state[0]
                for k, q, p, K in reads:
                    out[k][q] = self.label_logits(h[p], K).cpu()
                i = j
        finally:
            for m in marked:
                m.sieve_prefix = None
        return out

    @torch.no_grad()
    def logits_many(self, requests, budget=16384, reads=None, reread_max=None, orders=None):
        """Raw logits (served option order) per question of many requests [(state, questions)], batched by token budget
        (rows x longest row <= budget). Returns [[logits per question] or Unsupported] per request. reads / reread_max /
        orders default to the release's serving policy (head.pt "serving"; single read, one order without one)."""
        reads = self.policy["reads"] if reads is None else reads
        reread_max = self.policy["reread_max"] if reread_max is None else reread_max
        orders = self.policy["orders"] if orders is None else orders
        jobs, results, once = [], [], []
        for ri, (state, questions) in enumerate(requests):
            st, ins, opts, meta = self.encode(state, questions)
            results.append({"meta": meta, "z": [None] * len(ins), "z2": [None] * len(ins), "rot": [0] * len(ins)})
            try:
                block0 = self.fmt.state(ins, st)
                short = len(block0) + sum(len(self.fmt.branch(i, o)) for i, o in zip(ins, opts)) <= reread_max
                if self.uses_state_once(len(block0), len(ins), reads, short):
                    sets = [("z", [(q, self.fmt.branch(ins[q], opts[q]), len(opts[q])) for q in range(len(ins))])]
                    if orders == 2:                             # second order, only where it is used (see below)
                        qd = list(questions.values())
                        sh = [len(o) // 2 if m["type"] == "choice" and q.get("permute") is not False and len(o) > 1 else 0
                              for m, q, o in zip(meta, qd, opts)]
                        if any(sh):
                            results[-1]["rot"] = sh
                            sets.append(("z2", [(q, self.fmt.branch(ins[q], opts[q][s:] + opts[q][:s]), len(opts[q]))
                                                for q, s in enumerate(sh) if s]))
                    n = len(block0) + max(len(b) for _, bs in sets for _, b, _ in bs)
                    if self.max_tokens and n > self.max_tokens:
                        raise Unsupported(f"a pass of {n:,} tokens is over the validated limit {self.max_tokens:,} "
                                          "(not truncated)")
                    once.append((ri, block0, sets))
                    continue
                us = self.units(st, ins, opts, reads, reread_max)
                rot = []
                if orders == 2:                                 # second order: rotated by n // 2 where it may be permuted
                    qd = list(questions.values())
                    perm = [m["type"] == "choice" and q.get("permute") is not False and len(o) > 1
                            for m, q, o in zip(meta, qd, opts)]
                    if any(perm):
                        sh = [len(o) // 2 if p else 0 for o, p in zip(opts, perm)]
                        results[-1]["rot"] = sh
                        rot = self.units(st, ins, [o[s:] + o[:s] for o, s in zip(opts, sh)], reads, reread_max)
                self.check_length(us + rot)
            except Unsupported as e:
                results[-1] = e
                continue
            jobs += [(ri, "z", u) for u in us] + [(ri, "z2", u) for u in rot]
        order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2][0]))
        batch = []
        def flush():
            for (ri, key, _), got in zip([jobs[j] for j in batch], self.run_units([jobs[j][2] for j in batch])):
                for q, z in got:
                    results[ri][key][q] = z
        for j in order:
            if batch and (len(batch) + 1) * len(jobs[j][2][0]) > budget:
                flush(); batch = []
            batch.append(j)
        if batch:
            flush()
        for ri, block0, sets in once:
            for key, got in self.state_once(block0, sets).items():
                for q, z in got.items():
                    results[ri][key][q] = z
        out = []
        for r in results:
            if isinstance(r, Exception):
                out.append(r); continue
            zs = []
            for q, z in enumerate(r["z"]):
                if r["z2"][q] is not None and r["rot"][q]:      # back to the served order, average log-probabilities
                    s, n = r["rot"][q], len(z)
                    back = torch.empty_like(z)
                    back[[(i + s) % n for i in range(n)]] = r["z2"][q]
                    z = (torch.log_softmax(z, -1) + torch.log_softmax(back, -1)) / 2
                zs.append(z)
            out.append(zs)
        return out

    def decide(self, state, questions, temperature=None, **kw):
        """{question id: {key: probability}} of one request (calibration temperatures unless temperature is given)."""
        got = self.logits_many([(state, questions)], **kw)[0]
        if isinstance(got, Exception):
            raise got
        _, _, meta = to_record(state, questions)
        tq = ([temperature] * len(meta) if temperature is not None else
              question_temperatures(self.temperatures, meta, self.temperature, state) or [self.temperature] * len(meta))
        return {m["id"]: dict(zip(m["keys"], torch.softmax(z / t, -1).tolist())) for m, z, t in zip(meta, got, tq)}
