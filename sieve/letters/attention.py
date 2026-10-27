"""Attention rule A for Gemma 4 backbones (sliding-window and global layers), as Sieve-31B-v3 was trained and is served.

Segments and positions per token instead of a query x key mask: segment 0 is the shared state (block 0), k >= 1 is one
question's branch, -1 is padding. In the global layers of a bidirectional model the state reads the whole state, a
branch the whole state and its whole branch, never another branch; the sliding-window layers stay causal with their
window in every case. Keys may include cached ones first (k_seg / k_pos), e.g. a state computed once and reused.
"""
import os

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

# Sliding-window backbones (Gemma 4): a long masked pass runs its queries in blocks of ATTN_BLOCK, each against only
# the keys its mask lets it see (the same attention; the masked kernel would otherwise compute every query x key pair,
# 16x the work of a 1,024-token window at 16k tokens). Passes with fewer than ATTN_BLOCK_MIN queries run as one call.
ATTN_BLOCK = int(os.environ.get("SIEVE_ATTN_BLOCK", "1024"))
ATTN_BLOCK_MIN = int(os.environ.get("SIEVE_ATTN_BLOCK_MIN", "4096"))


def sdpa_blocks(query, key, value, mask, dropout, scale):
    """scaled_dot_product_attention(query, key, value, mask) with the queries in blocks: a block reads only the key
    columns where its mask has a True (every row keeps at least one, its own position)."""
    out = []
    for a in range(0, query.shape[2], ATTN_BLOCK):
        m = mask[:, :, a:a + ATTN_BLOCK]
        keys = m.any(dim=(0, 1, 2)).nonzero().squeeze(1)
        out.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, a:a + ATTN_BLOCK].contiguous(), key.index_select(2, keys), value.index_select(2, keys),
            attn_mask=m.index_select(3, keys), dropout_p=dropout, scale=scale))
    return torch.cat(out, 2)


class SegMask:
    """A compact attention mask for marked layers, from per-token segment and position ids instead of a query x key
    tensor (built per query block in sdpa_sieve, so a 54k-token pass never holds a 54k x 54k mask).

    Segments: 0 = the state, k >= 1 = a question's branch, -1 = padding. A real query sees a real key in the state or
    in its own segment; causal (bidir=False): only keys at its position or before, and with `window` only the last
    `window` positions (the sliding layers); bidir=True (global layers of a bidirectional model): every key of the
    state and of its own segment, so the state sees the whole state and a branch the whole state and the whole
    branch, never another branch, and the state never a branch. A padding query sees the state (never empty; its
    output is not used). q_* are [B, Lq], k_* are [B, Lk] (Lk = the keys the layer attends to, cached ones first)."""

    def __init__(self, q_seg, q_pos, k_seg, k_pos, window=None, bidir=False):
        self.q_seg, self.q_pos, self.k_seg, self.k_pos = q_seg, q_pos, k_seg, k_pos
        self.window, self.bidir = window, bidir

    def dense(self, a=0, b=None):
        """The [B, 1, b - a, Lk] boolean mask of queries a..b."""
        qs, qp = self.q_seg[:, a:b, None], self.q_pos[:, a:b, None]
        ks, kp = self.k_seg[:, None, :], self.k_pos[:, None, :]
        ok = (ks >= 0) & ((ks == 0) | (ks == qs))
        if not self.bidir:
            ok &= kp <= qp
            if self.window:
                ok &= (qp - kp) < self.window
        return (ok | ((qs < 0) & (ks == 0)))[:, None]


def sdpa_segmask(query, key, value, spec, dropout, scale):
    """Attention under a SegMask: one call for a short pass; queries in blocks of ATTN_BLOCK, each against only the
    keys it may see, for a long one."""
    Lq = query.shape[2]
    if Lq < ATTN_BLOCK_MIN:
        return torch.nn.functional.scaled_dot_product_attention(query.contiguous(), key, value, attn_mask=spec.dense(),
                                                                dropout_p=dropout, scale=scale)
    out = []
    for a in range(0, Lq, ATTN_BLOCK):
        m = spec.dense(a, a + ATTN_BLOCK)
        keys = m.any(dim=(0, 1, 2)).nonzero().squeeze(1)
        if len(keys) == key.shape[2]:                   # every key visible (a bidirectional block): no gather copy
            k, v = key, value
        else:
            k, v, m = key.index_select(2, keys), value.index_select(2, keys), m.index_select(3, keys)
        if bool(m.all()):                               # every query sees every key: no mask, the fastest kernel
            m = None
        out.append(torch.nn.functional.scaled_dot_product_attention(
            query[:, :, a:a + ATTN_BLOCK].contiguous(), k, v, attn_mask=m, dropout_p=dropout, scale=scale))
    return torch.cat(out, 2)


def seg_masks(seg, pos, window, bidir, k_seg=None, k_pos=None, k_seg_sliding=None, k_pos_sliding=None):
    """The {layer type: SegMask} of one pass for model(attention_mask=...): queries seg / pos [B, Lq]; keys the same
    tokens unless given (k_*: the global layers' keys, k_*_sliding: the sliding layers' keys, cached ones first)."""
    k_seg = seg if k_seg is None else k_seg
    k_pos = pos if k_pos is None else k_pos
    out = {"full_attention": SegMask(seg, pos, k_seg, k_pos, bidir=bidir)}
    if window:
        out["sliding_attention"] = SegMask(seg, pos, k_seg if k_seg_sliding is None else k_seg_sliding,
                                           k_pos if k_pos_sliding is None else k_pos_sliding, window=window)
    return out


def sdpa_sieve(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    """transformers' "sdpa" attention. Layers marked by mark_attention (a sliding-window Sieve backbone) run a masked
    pass with contiguous q / k / v, in query blocks when it is long: with a mask, PyTorch 2.8's memory-efficient kernel
    gets some lengths wrong for the strided q / k / v that transformers passes (Gemma 4's sliding layers at 1,025 /
    1,153 / 1,281 / 1,409 / 1,537 tokens: the last position off by ~4, fp32 and bf16, A100), and contiguous inputs are
    right at every length. With a mask transformers' path is exactly this (is_causal is off). Every other layer, and
    every unmasked pass, runs transformers' own function unchanged."""
    if not getattr(module, "sieve_attention", False) or attention_mask is None or kw.get("position_bias") is not None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kw)
    cap = getattr(module, "sieve_capture", None)    # a state pass: keep its keys / values for the branches after it
    if cap is not None:
        if getattr(module, "is_sliding", False):      # only the last window - 1 keys reach a branch: a copy, so the
            k0 = key.shape[2] - cap["sliding_keys"]   # rest of the state's keys are freed
            cap["kv"][module] = (key[:, :, k0:].clone(), value[:, :, k0:].clone())
        else:
            cap["kv"][module] = (key, value)
    pre = getattr(module, "sieve_prefix", None)     # a branch pass: the state's keys / values come first
    if pre is not None:
        key, value = torch.cat([pre[0], key], 2), torch.cat([pre[1], value], 2)
    n = getattr(module, "num_key_value_groups", 1)
    key, value = repeat_kv(key, n).contiguous(), repeat_kv(value, n).contiguous()
    if isinstance(attention_mask, SegMask):
        assert attention_mask.k_seg.shape[1] == key.shape[2], (attention_mask.k_seg.shape, key.shape)
        out = sdpa_segmask(query, key, value, attention_mask, dropout, scaling)
    elif attention_mask.dtype == torch.bool and query.shape[2] >= ATTN_BLOCK_MIN:
        out = sdpa_blocks(query, key, value, attention_mask, dropout, scaling)
    else:
        out = torch.nn.functional.scaled_dot_product_attention(query.contiguous(), key, value, attn_mask=attention_mask,
                                                               dropout_p=dropout, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def mark_attention(lm):
    """Route lm's attention layers through sdpa_sieve (transformers' "sdpa" entry becomes sdpa_sieve, which leaves the
    layers of every other model as they are). Returns how many layers were marked."""
    AttentionInterface.register("sdpa", sdpa_sieve)
    n = 0
    for mod in lm.modules():
        if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
            mod.sieve_attention = True; n += 1
    return n

