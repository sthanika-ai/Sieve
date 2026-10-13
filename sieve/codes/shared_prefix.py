"""State-once / isolated-branch forward.

The state (prefix) runs once; each question branch continues from what every decoder layer left at the end of the
state -- attention keys/values, and for Gated DeltaNet layers the short-conv window and the recurrent state -- with
positions continuing from the prefix end. A branch attends to its record's prefix and to itself, never to a sibling.

Modified from Kev (https://github.com/jaredpalmer/kev, Apache-2.0, (c) Jared Palmer). Covers:
  * attention-only Qwen3 as well as hybrid Qwen3.5 / Qwen3.8 (Gated DeltaNet + gated attention);
  * Gemma-4 (sliding-window + global attention layers, per-type RoPE; serving path only): SDPA ignores a layer's
    sliding window unless the mask carries it, so every sliding layer gets an explicit window mask once a sequence is
    longer than the window; long states are prefilled in chunks, and sliding layers keep only their last `window`
    keys for the branches;
  * a serving path (`encode_prefix` + `run_branches`) that keeps the per-layer prefix state, can be cached across
    requests, and runs any number of branches in memory-bounded chunks without re-running the state;
  * an optional block-bidirectional mask (serving path): state tokens see the whole state, a branch's
    tokens see the whole state and their whole branch, branches never see each other. Gated DeltaNet layers stay left
    to right; sliding layers look `window` tokens both ways.
Each branch matches running (state + branch) as its own sequence.
"""
from __future__ import annotations

import inspect
import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_PREFILL_CHUNK = 4096        # Gemma-4: states longer than this are prefilled in chunks (bounded mask memory)


class Prefix:
    """One decoder layer's view of the shared state, duck-typing the cache calls Qwen3 / Qwen3.5 / Gemma-4 layers make.
    The state pass records what the layer leaves; the branch pass reads it. Nothing is written in place, so autograd
    sees every branch's gradient flow back into the state.

    `window` (sliding-attention layers): only the last `window` positions are kept for later passes; `offset` is the
    absolute position of the first kept key. `appending` (chunked prefill): each update extends the stored keys."""
    record_past = True            # keeps one-token passes off the in-place decode kernel

    def __init__(self, window: int | None = None):
        self.keys = self.values = self.conv = None
        self.recurrent_states = {}
        self.window, self.offset, self.appending = window, 0, False

    # --- cache protocol -------------------------------------------------------------------------------------------
    @property
    def layers(self):
        return self

    def __getitem__(self, layer_idx):
        return self

    def has_previous_state(self, layer_idx=None, state_idx=0):
        return self.conv is not None

    def get_seq_length(self, layer_idx=0):
        return 0 if self.keys is None else self.offset + self.keys.shape[-2]

    def update_conv_state(self, conv_input, layer_idx, conv_kernel_size, **kwargs):
        if self.conv is not None:
            return torch.cat([self.conv, conv_input], -1)
        self.conv = F.pad(conv_input, (max(0, conv_kernel_size - conv_input.shape[-1]), 0))[..., -conv_kernel_size:]
        return conv_input

    def update_recurrent_state(self, recurrent_state, layer_idx, **kwargs):
        self.recurrent_states.setdefault(0, recurrent_state)
        return recurrent_state

    def _store(self, keys, values, end: int):
        """Keep keys/values covering absolute positions [end - n, end); a windowed layer keeps the last `window`."""
        if self.window is not None and keys.shape[-2] > self.window:
            keys, values = keys[..., -self.window:, :], values[..., -self.window:, :]
        self.keys, self.values, self.offset = keys, values, end - keys.shape[-2]

    def update(self, keys, values, layer_idx=None, *args, **kwargs):
        if self.keys is None:
            self._store(keys, values, keys.shape[-2])
            return keys, values                                  # this pass sees all of its own keys (mask applies)
        k, v = torch.cat([self.keys, keys], -2), torch.cat([self.values, values], -2)
        if self.appending:
            self._store(k, v, self.offset + k.shape[-2])
        return k, v

    # --- fan-out ----------------------------------------------------------------------------------------------------
    def _like(self) -> "Prefix":
        out = Prefix(self.window)
        out.offset = self.offset
        return out

    def gathered(self, owner: torch.Tensor) -> "Prefix":
        """A new Prefix holding row owner[i] of this one for branch i (training: several records per batch)."""
        out = self._like()
        pick = lambda t: None if t is None else t.index_select(0, owner)
        out.keys, out.values, out.conv = pick(self.keys), pick(self.values), pick(self.conv)
        out.recurrent_states = {k: pick(v) for k, v in self.recurrent_states.items()}
        return out

    def expanded(self, n: int) -> "Prefix":
        """Serving: one cached state fanned out to n branches (views; materialised only inside the layer)."""
        out = self._like()
        ex = lambda t: None if t is None else t.expand(n, *t.shape[1:])
        out.keys, out.values, out.conv = ex(self.keys), ex(self.values), ex(self.conv)
        out.recurrent_states = {k: ex(v).contiguous() for k, v in self.recurrent_states.items()}
        return out

    def nbytes(self) -> int:
        ts = [self.keys, self.values, self.conv, *self.recurrent_states.values()]
        return sum(t.numel() * t.element_size() for t in ts if t is not None)


# ------------------------------------------------------------------------------------------------------ helpers
def text_model(backbone):
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    if hasattr(base, "language_model"):          # Qwen3_5Model / Gemma4Model (multimodal wrappers)
        base = base.language_model
    return base


def _kind(base, i: int) -> str:
    lt = getattr(base.config, "layer_types", None)
    k = lt[i] if lt else "full_attention"
    return k if k in ("linear_attention", "sliding_attention") else "full_attention"


def _is_mrope(base) -> bool:
    return base.config.model_type.startswith("qwen3_5")


def _is_gemma4(base) -> bool:
    return base.config.model_type.startswith("gemma4")


def _window(base) -> int | None:
    lt = getattr(base.config, "layer_types", None) or []
    return base.config.sliding_window if "sliding_attention" in lt else None


def _check_supported(base):
    if getattr(base.config, "num_kv_shared_layers", 0):
        raise NotImplementedError("Gemma-4 cross-layer KV sharing (E2B/E4B) is not supported by the state-once graph")
    if getattr(base.config, "hidden_size_per_layer_input", 0):
        raise NotImplementedError("Gemma-4 per-layer embeddings (E2B/E4B) are not supported by the state-once graph")


def _rope(base, h, pos):
    return base.rotary_emb(h, pos[None].expand(3, -1, -1) if _is_mrope(base) else pos)


def _ropes(base, h, pos) -> dict:
    """Position embeddings per layer kind (Gemma-4 has one RoPE per attention type; Qwen one for every layer)."""
    if _is_gemma4(base):
        return {k: base.rotary_emb(h, pos, k) for k in sorted(set(base.config.layer_types))}
    r = _rope(base, h, pos)
    return {"full_attention": r, "linear_attention": r, "sliding_attention": r}


def _attn_mask(allow, dtype, attn_impl):
    allow = allow[:, None]
    if attn_impl == "sdpa":
        return allow
    return torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)


def _allow(qpos, kpos, kreal, kind: str, window: int | None, bidir: bool):
    """qpos [B, Lq], kpos [B, Lk] absolute positions, kreal [B, Lk] -> [B, Lq, Lk] who may attend to whom. A token may
    always attend to itself (d == 0), so padded rows never go all-masked (no NaN leaks through the values)."""
    d = qpos[:, :, None] - kpos[:, None, :]
    allow = kreal[:, None, :] if bidir else kreal[:, None, :] & (d >= 0)
    if kind == "sliding_attention" and window is not None:
        allow = allow & (d < window) & (d > -window if bidir else True)
    return allow | (d == 0)


def _layer_call(layer, h, rope, mask, pos, cache):
    kw = {}
    if getattr(layer, "_sieve_kv_states", None) is None:
        layer._sieve_kv_states = "shared_kv_states" in inspect.signature(layer.forward).parameters
    if layer._sieve_kv_states:
        kw["shared_kv_states"] = {}          # Gemma-4 writes its last full-length K/V here (read only by KV-shared layers)
    return layer.forward(h, position_embeddings=rope, attention_mask=mask, position_ids=pos, past_key_values=cache, **kw)


def _method(module, name, fn):
    if not hasattr(module, name):
        setattr(module, name, types.MethodType(fn, module))
    return getattr(module, name)


# ------------------------------------------------------------------------------------------------- training path
def forward_shared(backbone, s_ids, s_pos, s_real, b_ids, b_pos, b_real, owner):
    """Several records at once. States left-padded [R, Ls]; branches right-padded [B, Lb]; owner[b] = record of branch
    b. Returns the final-normed branch hidden states [B, Lb, d] (model dtype). Gradient checkpointing, when enabled on
    the backbone, recomputes each layer's (state pass + branch pass) as one unit."""
    base = text_model(backbone)
    if _window(base) is not None:
        raise NotImplementedError("the training path does not handle sliding-window layers yet (Gemma-4: serving only)")
    device = s_ids.device
    Ls, Lb = s_ids.shape[1], b_ids.shape[1]
    h_s = base.embed_tokens(s_ids)
    h_b = base.embed_tokens(b_ids)
    all_real = bool(s_real.all())
    if not all_real:
        h_s = h_s * s_real[..., None]
    h_b = h_b * b_real[..., None]
    dtype, attn = h_s.dtype, base.config._attn_implementation
    causal = lambda n: torch.ones(n, n, dtype=torch.bool, device=device).tril()
    eye = lambda n: torch.eye(n, dtype=torch.bool, device=device)
    if all_real:          # unpadded states: SDPA's causal kernel, no mask tensor
        m_s = {"full_attention": None, "linear_attention": None}
    else:
        allow_s = (causal(Ls)[None] & s_real[:, None, :]) | eye(Ls)[None]
        m_s = {"full_attention": _attn_mask(allow_s, dtype, attn), "linear_attention": s_real.to(dtype)}
    allow_b = torch.cat([s_real[owner][:, None, :].expand(-1, Lb, -1),
                         (causal(Lb)[None] & b_real[:, None, :]) | eye(Lb)[None]], -1)
    m_b = {"full_attention": _attn_mask(allow_b, dtype, attn), "linear_attention": b_real.to(dtype)}
    rope_s, rope_b = _rope(base, h_s, s_pos), _rope(base, h_b, b_pos)
    recompute = getattr(base, "gradient_checkpointing", False) and base.training and torch.is_grad_enabled()
    for i, layer in enumerate(base.layers):
        k = _kind(base, i)
        step = _method(layer, "sieve_shared_step", _step)
        args = (h_s, h_b, rope_s, rope_b, m_s[k], m_b[k], s_pos, b_pos, s_real, owner, all_real)
        h_s, h_b = checkpoint(step, *args, use_reentrant=False) if recompute else step(*args)
    return base.norm(h_b)


def _step(layer, h_s, h_b, rope_s, rope_b, m_s, m_b, s_pos, b_pos, s_real, owner, all_real):
    prefix = Prefix()
    h_s = _layer_call(layer, h_s, rope_s, m_s, s_pos, prefix)
    h_b = _layer_call(layer, h_b, rope_b, m_b, b_pos, prefix.gathered(owner))
    return (h_s if all_real else h_s * s_real[..., None]), h_b


# -------------------------------------------------------------------------------------------------- serving path
def _state_masks(base, L: int, device, dtype, attn, bidir: bool, q0: int = 0, k0: dict | None = None) -> dict:
    """Masks for state queries at positions [q0, q0 + n) (n = L - q0) over keys from k0[kind] (default 0) to L.
    None where SDPA's own causal kernel is exact (no window cut, no bidirectional reading)."""
    window, out = _window(base), {"linear_attention": None}
    for kind in ("full_attention",) + (("sliding_attention",) if window is not None else ()):
        start = (k0 or {}).get(kind, 0)
        if not bidir and q0 == 0 and start == 0 and (kind == "full_attention" or L <= window):
            out[kind] = None
            continue
        if bidir and q0 == 0 and start == 0 and (kind == "full_attention" or L <= window):
            out[kind] = _attn_mask(torch.ones(1, 1, L, dtype=torch.bool, device=device), dtype, attn)   # broadcast
            continue
        qpos = torch.arange(q0, L, device=device)[None]
        kpos = torch.arange(start, L, device=device)[None]
        allow = _allow(qpos, kpos, torch.ones_like(kpos, dtype=torch.bool), kind, window, bidir)
        out[kind] = _attn_mask(allow, dtype, attn)
    return out


@torch.inference_mode()
def encode_prefix(backbone, ids: torch.Tensor, bidir: bool = False, chunk: int | None = DEFAULT_PREFILL_CHUNK) -> list[Prefix]:
    """ids [1, Ls] -> per-layer Prefix (the whole state, run once, unpadded). Causal by default; `bidir`: the state
    reads itself both ways (attention layers). Gemma-4 states longer than `chunk` are prefilled chunk by chunk."""
    base = text_model(backbone)
    _check_supported(base)
    window = _window(base)
    kinds = [_kind(base, i) for i in range(len(base.layers))]
    out = [Prefix(window if k == "sliding_attention" else None) for k in kinds]
    L = ids.shape[1]
    dtype, attn = base.embed_tokens.weight.dtype, base.config._attn_implementation
    if chunk and L > chunk and not bidir and _is_gemma4(base):
        for p in out:
            p.appending = True
        for s in range(0, L, chunk):
            e = min(L, s + chunk)
            pos = torch.arange(s, e, device=ids.device)[None]
            h = base.embed_tokens(ids[:, s:e])
            ropes = _ropes(base, h, pos)
            k0 = {"full_attention": 0, "sliding_attention": max(0, s - window) if window else 0}
            masks = _state_masks(base, e, ids.device, dtype, attn, False, q0=s, k0=k0) if s else \
                _state_masks(base, e, ids.device, dtype, attn, False)
            for layer, p, k in zip(base.layers, out, kinds):
                h = _layer_call(layer, h, ropes[k], masks[k], pos, p)
        for p in out:
            p.appending = False
        return out
    pos = torch.arange(L, device=ids.device)[None]
    h = base.embed_tokens(ids)
    ropes = _ropes(base, h, pos)
    masks = _state_masks(base, L, ids.device, dtype, attn, bidir)
    for layer, p, k in zip(base.layers, out, kinds):
        h = _layer_call(layer, h, ropes[k], masks[k], pos, p)
    return out


@torch.inference_mode()
def run_branches(backbone, prefixes: list[Prefix], Ls: int, b_ids, b_real, gather_idx, bidir: bool = False):
    """Branches [B, Lb] (right-padded) of ONE cached state. gather_idx [B, G] positions to return. -> [B, G, d]."""
    base = text_model(backbone)
    device = b_ids.device
    B, Lb = b_ids.shape
    window = _window(base)
    b_pos = (Ls + torch.arange(Lb, device=device))[None].expand(B, -1)
    h = base.embed_tokens(b_ids) * b_real[..., None]
    dtype, attn = h.dtype, base.config._attn_implementation
    masks = {"linear_attention": b_real.to(dtype)}
    kinds = [_kind(base, i) for i in range(len(base.layers))]
    for kind in ("full_attention", "sliding_attention"):
        p = next((q for q, k in zip(prefixes, kinds) if k == kind), None)
        if p is None:
            continue
        kept = p.keys.shape[-2] if p.keys is not None else 0
        ppos = torch.arange(p.offset, p.offset + kept, device=device)[None].expand(B, -1)
        kpos = torch.cat([ppos, b_pos], -1)
        kreal = torch.cat([torch.ones(B, kept, dtype=torch.bool, device=device), b_real], -1)
        masks[kind] = _attn_mask(_allow(b_pos, kpos, kreal, kind, window, bidir), dtype, attn)
    ropes = _ropes(base, h, b_pos)
    for layer, p, k in zip(base.layers, prefixes, kinds):
        h = _layer_call(layer, h, ropes[k], masks[k], b_pos, p.expanded(B))
    h = base.norm(h)
    return torch.gather(h, 1, gather_idx[..., None].expand(-1, -1, h.shape[-1]))


def force_torch_kernels():
    """Tests on CPU: route Qwen3.5's linear-attention kernels to the torch reference even when fla is installed."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(m, name, None)
        while fn is not None and hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        if fn is not None:
            setattr(m, name, fn)
