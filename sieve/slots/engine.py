"""Decision Index engine for Sieve-31B and Sieve-31B-v2: google/gemma-4-31B-it + the release's weights, read at answer slots.

- Sieve-31B: a LoRA adapter, applied unmerged. Sieve-31B-v2: a full fine-tune of the decoder layers (stored as an adapter,
  loaded in place) with non-causal global layers, as trained. sieve/slots/readout.py reads both from the release's files.

- Prompts and batching come from the decider-ai package (pip, version 1.6.0), used unmodified: its chat layout puts one
  user turn in the model's own chat template with thinking switched off and one "Answer: (" slot per question; options
  keep the order the request gives them.
- The model computation is sieve/slots/readout.py: option-label logits at each slot, softmax over the question's options at
  the fixed temperature 8.5495 (the training temperature).
- Nothing is generated, truncated or filtered; no per-benchmark settings. A request with a row longer than `max_tokens`
  or more than 255 options is refused as Unsupported.
- A CUDA OOM halves the forward chunk and retries; a single row that still does not fit raises RuntimeError (an error
  row is retried on resume).

Run:
  python -m decision_index pipeline --engine sieve.decision_index_engine:SieveEngine \
      --option model=sthanika-ai/Sieve-31B --out runs/Sieve-31B
"""
import json

import torch

try:
    import decider.infer as DI
    from decider import temperature as TT
    from decider.infer import Decider
    from decider.prompt import MAX_OPTIONS, chat_template
    from decider.systemone import assemble, unique_tokens
except ImportError as e:
    raise ImportError("Sieve-31B builds its prompts with the decider-ai package; install it without its dependencies: "
                      "pip install --no-deps decider-ai==1.6.0") from e

try:
    from decision_index.engines.base import Engine, Unsupported
except ImportError:                                  # the kit is only needed to run the Decision Index
    from .._kit import Engine, Unsupported

from .readout import make_readout

BASE = "google/gemma-4-31B-it"
BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
TEMPERATURE = 8.5495
ADAPTER = "sthanika-ai/Sieve-31B"


class SlotsEngine(Engine):
    name = "sieve-31b"
    latency = "In-process request wall time (prompt construction + forward); one request at a time; excludes model loading."

    def __init__(self, model=ADAPTER, base=BASE, revision=BASE_REVISION, device="cuda", temperature=TEMPERATURE,
                 max_tokens=32768, fwd_tokens=16384, split=None, **options):
        super().__init__(model=model, base=base, revision=revision, device=device, temperature=temperature,
                         max_tokens=max_tokens, fwd_tokens=fwd_tokens, split=split, **options)
        adapter = model                    # the Sieve-31B adapter: a Hub repo id or a local folder
        budget = None
        if split not in (None, "", False):
            # split: per-GPU memory budget when one GPU is too small, e.g. '{"0": "40GiB", "1": "40GiB"}'
            raw = split if isinstance(split, dict) else json.loads(split)
            budget = {int(k): v for k, v in raw.items()}
        original = DI.DecisionModel        # the orchestrator builds its model object by this name; ours is SlotReadout
        DI.DecisionModel = make_readout(max_memory=budget, adapter=adapter, revision=revision)
        try:
            self.d = Decider(base, device=device, dtype=torch.bfloat16, temperature=float(temperature), use_graphs=False)
        finally:
            DI.DecisionModel = original
        self.d.layout = "chat"             # = decider.serve's DECIDER_LAYOUT=chat (the stock checkpoint has no decider_config)
        self.d.chat = chat_template(self.d.m.tok)
        self.max_tokens = int(max_tokens)
        self.fwd_tokens = int(fwd_tokens)
        m = self.d.m
        self.model_id = adapter
        self.provenance = {
            "kind": "full fine-tune (decoder layers, stored as an adapter)" if m.full_release else "LoRA adapter",
            "attention": m.attention,
            "base": f"{base}@{revision}",
            "adapter": adapter,
            "policy": "base model + the release's weights in its chat template with thinking off, one answer slot per question, "
                      "softmax of the option-label logits at a fixed temperature; independent rows; rows over max_tokens "
                      "are refused, never truncated",
            "temperature": float(temperature),
            "max_tokens": self.max_tokens,
        }

    def runtime(self):
        dm = getattr(self.d.m.lm, "hf_device_map", None)
        return {"chat_head": self.d.chat.head_text, "chat_tail": self.d.chat.tail_text, "dtype": "bfloat16",
                "devices": sorted({str(v) for v in dm.values()}) if dm else None}

    def __call__(self, state, questions):
        for q in questions.values():
            if q["type"] == "choice" and len(q["criteria"]) > MAX_OPTIONS:
                raise Unsupported(f"{len(q['criteria'])} options; the label head has {MAX_OPTIONS}")
        try:
            # max_state_tokens is set past any real state so the state is never cut; the row check below refuses instead.
            rqs, index, items = self.d._system_one_items(state, questions, True, 1 << 30, "state_first")
        except ValueError as e:            # malformed question (e.g. noul without a described criterion)
            raise Unsupported(str(e))
        longest = max(len(it["ids"]) for it in items)
        if longest > self.max_tokens:
            raise Unsupported(f"a row has {longest} tokens; the context limit is {self.max_tokens}")
        T = TT.for_items(self.d.T, self.d.T_by_type, items)
        budget = self.fwd_tokens
        while True:
            try:
                flatp = self.d._system_one_probs(items, "state_first", budget, T)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if budget <= longest:
                    raise RuntimeError(f"CUDA OOM on a single {longest}-token row")
                budget = max(longest, budget // 2)
        resp = {"model": self.model_id, "answers": assemble(rqs, index, flatp),
                "usage": {"input_tokens": unique_tokens(items), "output_tokens": 0}}
        return resp, {"T": float(self.d.T), "index": [list(x) for x in index], "row_probs": flatp}
