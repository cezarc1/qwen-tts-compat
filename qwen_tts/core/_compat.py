# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bridge the Transformers 4.57.x and 5.x model-authoring APIs.

The Qwen3-TTS modeling code is written against Transformers' internal authoring
API, and upstream therefore hard-pins ``transformers==4.57.3``. Four of those
internals changed in Transformers 5.x. This module adapts each one so a single
source tree runs on both; every helper is a pass-through on 4.57.x.

The adapters deliberately delegate to Transformers' own implementations rather
than reimplementing them, so attention backends (FlashAttention 2, flex
attention, SDPA) keep working exactly as the installed release intends.
"""

from __future__ import annotations

import torch
import transformers
from transformers.masking_utils import create_causal_mask as _create_causal_mask
from transformers.masking_utils import (
    create_sliding_window_causal_mask as _create_sliding_window_causal_mask,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils.generic import check_model_inputs as _check_model_inputs

#: True when running on Transformers 5.x or newer.
IS_TRANSFORMERS_V5 = int(transformers.__version__.split(".", 1)[0]) >= 5


def check_model_inputs():
    """Return the ``check_model_inputs`` decorator.

    4.57.x exposes a decorator *factory*, so call sites read
    ``@check_model_inputs()``. 5.x exposes the decorator itself. Calling this
    wrapper yields the decorator either way, leaving call sites unchanged.
    """
    return _check_model_inputs if IS_TRANSFORMERS_V5 else _check_model_inputs()


def default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None, **kwargs):
    """Compute inverse frequencies for the plain (unscaled) RoPE.

    Transformers 5.x dropped the ``"default"`` entry from
    ``ROPE_INIT_FUNCTIONS``; this reproduces the 4.57.x implementation.
    """
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def rope_init_fn(rope_type: str):
    """Resolve a RoPE initializer, without mutating Transformers' global registry."""
    if rope_type in ROPE_INIT_FUNCTIONS:
        return ROPE_INIT_FUNCTIONS[rope_type]
    if rope_type == "default":
        return default_rope_parameters
    raise KeyError(f"unknown rope_type {rope_type!r}")


def pad_token_id_of(config):
    """Return ``config.pad_token_id``, or ``None`` when the base config omits it.

    4.57.x's ``PretrainedConfig`` always defined ``pad_token_id``, defaulting it
    to ``None``; 5.x removed the attribute. ``None`` is what the Qwen3-TTS
    configs actually resolve to on 4.57.x, and the value reaches ``nn.Embedding``
    as ``padding_idx`` — so preserving ``None``, rather than substituting a codec
    id, keeps the embedding tables identical across versions.
    """
    return getattr(config, "pad_token_id", None)


def ensure_cache_position(cache_position, past_key_values, *, length: int, device):
    """Recreate ``cache_position`` when Transformers 5.x does not supply it.

    4.57.x threaded ``cache_position`` through model forwards during
    generation. 5.x derives positions from the cache instead and stops passing
    it — the same change that made ``create_causal_mask``'s ``cache_position``
    parameter "deprecated and unused".

    Qwen3-TTS's talker uses it to choose between its prefill and decode
    position-id branches. Left as ``None``, every decode step re-runs the
    *prefill* branch and rebuilds RoPE over the whole sequence, so a
    single-token step emits full-sequence cos/sin and the attention output no
    longer matches ``input_shape``.
    """
    if cache_position is not None:
        return cache_position
    seen = 0
    if past_key_values is not None:
        try:
            seen = int(past_key_values.get_seq_length())
        except (AttributeError, TypeError):
            seen = 0
    return torch.arange(seen, seen + length, device=device)


def _adapt_mask_kwargs(kwargs: dict) -> dict:
    """Translate 4.57.x mask keyword arguments to their 5.x spellings."""
    if not IS_TRANSFORMERS_V5:
        return kwargs
    kwargs = dict(kwargs)
    if "input_embeds" in kwargs:
        # Renamed to `inputs_embeds` in 5.x.
        kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
    # 5.x derives cache positions internally and documents the parameter as
    # "deprecated and unused".
    kwargs.pop("cache_position", None)
    return kwargs


def create_causal_mask(**kwargs):
    """Version-agnostic :func:`transformers.masking_utils.create_causal_mask`."""
    return _create_causal_mask(**_adapt_mask_kwargs(kwargs))


def create_sliding_window_causal_mask(**kwargs):
    """Version-agnostic :func:`transformers.masking_utils.create_sliding_window_causal_mask`."""
    return _create_sliding_window_causal_mask(**_adapt_mask_kwargs(kwargs))
