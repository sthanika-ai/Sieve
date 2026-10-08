"""CUDA-graph replay of the question pass for short requests, which removes per-kernel launch overhead.

One graph is captured per shape bucket (questions, branch length, state length). Requests larger than
`max_graph_tokens` use the eager path.
"""
import torch

from .encode import rows_of


def _bucket(x, buckets):
    for b in buckets:
        if b >= x:
            return b
    return None


class CudaGraphRunner:
    Q_BUCKETS = (1, 2, 4, 8, 16)
    LB_BUCKETS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    LS_BUCKETS = (32, 64, 128, 192, 256, 320, 384, 512, 768, 1024)

    def __init__(self, m, max_graph_tokens=1024):
        import transformers
        if int(transformers.__version__.split(".")[0]) < 5:
            raise RuntimeError("CUDA graphs need transformers >= 5")
        self.m, self.max_graph_tokens = m, max_graph_tokens
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()

    def _static_cache(self, Qb, Lsb):
        m = self.m
        ids = torch.full((1, Lsb), m.pad_id, dtype=torch.long, device=m.device)
        with torch.no_grad():
            out = m.lm(input_ids=ids, past_key_values=m._new_cache(), use_cache=True)
        cache = m._replicate(out.past_key_values, Qb)
        attn, lin = {}, {}
        for i, l in enumerate(cache.layers):
            if hasattr(l, "recurrent_states"):
                lin[i] = (l.conv_states[0], l.recurrent_states[0])
            else:
                attn[i] = (l.keys, l.values)
        return cache, attn, lin

    def _reset(self, g):
        # a forward pass re-binds the attention layers' keys/values, so point them back at the static buffers
        for i, (k, v) in g["attn"].items():
            g["cache"].layers[i].keys, g["cache"].layers[i].values = k, v

    def _capture(self, Qb, Lbb, Lsb):
        m = self.m
        cache, attn, lin = self._static_cache(Qb, Lsb)
        g = {"cache": cache, "attn": attn, "lin": lin,
             "ids": torch.full((Qb, Lbb), m.pad_id, dtype=torch.long, device=m.device),
             "pos": torch.zeros((Qb, Lbb), dtype=torch.long, device=m.device),
             "mask": torch.ones((Qb, 1, Lbb, Lsb + Lbb), dtype=torch.bool, device=m.device),
             "lin_init": {i: (c.clone(), r.clone()) for i, (c, r) in lin.items()}}
        g["mask"][:, :, :, Lsb:] = torch.tril(torch.ones(Lbb, Lbb, dtype=torch.bool, device=m.device))
        masks = {"full_attention": g["mask"], "linear_attention": None}

        def run():
            self._reset(g)
            for i, (c, r) in g["lin"].items():
                c.copy_(g["lin_init"][i][0])
                r.copy_(g["lin_init"][i][1])
            return m.lm(input_ids=g["ids"], position_ids=g["pos"], attention_mask=masks,
                        past_key_values=g["cache"], use_cache=True).last_hidden_state

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            self._reset(g)
            with torch.cuda.graph(graph, pool=self.pool):
                g["out"] = m.lm(input_ids=g["ids"], position_ids=g["pos"], attention_mask=masks,
                                past_key_values=g["cache"], use_cache=True).last_hidden_state
        g["graph"] = graph
        self._reset(g)
        return g

    @torch.no_grad()
    def logits(self, rec, prefix, temperature=None):
        m = self.m
        Ls = rec["n_state"]
        _, _, rows = rows_of(rec)
        Q, Lb = len(rows), max(len(r["ids"]) for r in rows)
        tq = temperature if isinstance(temperature, (list, tuple)) else [temperature] * Q
        Qb, Lbb, Lsb = _bucket(Q, self.Q_BUCKETS), _bucket(Lb, self.LB_BUCKETS), _bucket(Ls, self.LS_BUCKETS)
        if Qb is None or Lbb is None or Lsb is None or Qb * Lbb > self.max_graph_tokens:
            return m.logits(rec, prefix=prefix, temperature=temperature)
        key = (Qb, Lbb, Lsb)
        if key not in self.graphs:
            self.graphs[key] = self._capture(*key)
        g = self.graphs[key]
        src = prefix["cache"]
        for i, l in enumerate(src.layers if hasattr(src, "layers") else []):
            if i in g["attn"]:
                k, v = g["attn"][i]
                k[:, :, :Ls].copy_(l.keys.expand(Qb, -1, -1, -1))
                v[:, :, :Ls].copy_(l.values.expand(Qb, -1, -1, -1))
            else:
                c, r = g["lin"][i]
                c.copy_(l.conv_states[0].expand_as(c))
                r.copy_(l.recurrent_states[0].expand_as(r))
        g["ids"].fill_(m.pad_id)
        g["pos"].zero_()
        mask = g["mask"]
        mask[:, :, :, :Lsb] = False
        mask[:, :, :, :Ls] = True
        for i, r in enumerate(rows):
            n = len(r["ids"])
            g["ids"][i, :n] = torch.tensor(r["ids"], device=m.device)
            g["pos"][i, :n] = torch.tensor(r["pos"], device=m.device)
            if n < Lbb:
                g["pos"][i, n:] = r["pos"][-1]
        g["graph"].replay()
        h = g["out"].to(m.head_dtype)
        out = []
        for i, r in enumerate(rows):
            idx = torch.tensor(r["opts"], device=m.device)
            out.append(m.head(h[i, r["decide"]], h[i, idx], tq[i]))
        return out
