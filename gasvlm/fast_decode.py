"""Fast decode path: static KV cache + CUDA-graph-compiled decode step.

GasVLM's language model is deep and narrow (32 layers, hidden 960), so at
batch size one the eager decode loop is dominated by kernel-launch overhead
rather than arithmetic, roughly 2,000 launches per generated token with the
GPU busy under 15% of the wall clock. Capturing the decode step as a CUDA
graph replays those launches as one unit and cuts VQA/GCG latency by ~4x on
a GH200.

Mechanics
---------
* A preallocated static KV cache with layers pinned to fp32 buffers. The
  visual prefix reaches the LM in fp32 because layer_norm upcasts under
  autocast, so prefill keys arrive fp32 while decode-step keys arrive bf16;
  pinning the buffers to fp32 and upcasting incoming states mirrors exactly
  what the dynamic cache's torch.cat promotion did implicitly, keeping
  prefill numerics unchanged.
* ``torch.compile(mode="reduce-overhead")`` on the inner causal LM, hooked
  through HF generate's ``get_compiled_call`` so only the fixed-shape decode
  step is captured; prefill stays eager.

Numerics
--------
The captured path is not bit-identical to the eager one: with a static
cache, sdpa reduces over the padded key length, which changes floating-point
summation order and can flip a near-tied greedy argmax. Only text generation
uses the captured path. ``generate_with_masks`` then decodes each [SEG] mask
from one eager teacher-forced forward pass over the prompt and the generated
tokens, so a mask changes only when its generated text changes. Results can
differ slightly from the eager path (``--no-fast-decode`` in eval.py). The
first call triggers compilation and takes about a minute; all subsequent
calls run at full speed.

Usage
-----
    from gasvlm.fast_decode import enable_fast_decode
    enable_fast_decode(model)          # after build_model(), before eval
"""
from __future__ import annotations

import torch

__all__ = ["enable_fast_decode"]


def enable_fast_decode(model, max_cache_len: int = 704) -> None:
    """Wrap ``model.llm.generate`` with the captured decode path in place.

    ``max_cache_len`` must cover prefill length + max_new_tokens; the default
    704 covers GasVLM's ~136-token prefill plus the 512-token GCG budget. A
    call that would not fit falls back to the eager dynamic-cache path, so
    enabling this is always safe.
    """
    from transformers.cache_utils import Cache, StaticLayer

    llm = model.llm

    class _CastStaticLayer(StaticLayer):
        """StaticLayer with fp32 buffers; upcasts incoming bf16 states."""

        def update(self, key_states, value_states, cache_kwargs=None):
            if not self.is_initialized:
                self.lazy_initialization(key_states.float())
            return StaticLayer.update(
                self, key_states.to(self.dtype), value_states.to(self.dtype),
                cache_kwargs)

    txt_cfg = llm.config.get_text_config(decoder=True)
    cache = Cache(layers=[_CastStaticLayer(max_cache_len=max_cache_len)
                          for _ in range(txt_cfg.num_hidden_layers)])

    # The PEFT wrapper delegates generate() to the inner causal LM, whose
    # get_compiled_call() HF generate consults when the cache is compileable.
    inner = llm.base_model.model if hasattr(llm, "base_model") else llm
    compiled = torch.compile(inner.__call__, mode="reduce-overhead",
                             fullgraph=False, backend="inductor")

    def _wrapped(*a, **kw):
        out = compiled(*a, **kw)
        # CUDA-graph outputs are overwritten on replay, so copy any hidden
        # states out of the graph's static buffers. generate_with_masks does
        # not request them (it reruns one eager forward for the [SEG] states);
        # this guards callers that pass output_hidden_states to generate.
        hs = getattr(out, "hidden_states", None)
        if hs is not None:
            out.hidden_states = tuple(h.clone() for h in hs)
        return out

    inner.get_compiled_call = lambda compile_config: _wrapped

    orig_generate = llm.generate

    def _fast_generate(*args, **kw):
        inputs_embeds = kw.get("inputs_embeds")
        max_new = kw.get("max_new_tokens", 128)
        fits = (inputs_embeds is not None
                and "past_key_values" not in kw
                and inputs_embeds.shape[1] + max_new <= max_cache_len)
        if not fits:
            return orig_generate(*args, disable_compile=True, **kw)
        cache.reset()
        return orig_generate(*args, past_key_values=cache, **kw)

    llm.generate = _fast_generate
