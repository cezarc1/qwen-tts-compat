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
"""Span the Transformers 5.x model-authoring API, which moved during the 5 series.

Three internals this model builds on changed at different points in 5.x:

===========================  ===========================================
``check_model_inputs``       replaced by ``merge_with_config_defaults`` in 5.1
``ROPE_INIT_FUNCTIONS``      lost ``"default"``; gained ``"proportional"`` in 5.6
``create_causal_mask``       renamed ``input_embeds`` -> ``inputs_embeds`` in
                             5.2, then dropped ``cache_position`` in 5.9
===========================  ===========================================

Rather than pin above all of it, each helper detects what the installed release
actually provides. Detection is by introspection, not version comparison, so
back-ports and forks behave correctly too.
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import torch
from transformers.masking_utils import create_causal_mask as _create_causal_mask
from transformers.masking_utils import (
    create_sliding_window_causal_mask as _create_sliding_window_causal_mask,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def check_model_inputs():
    """Return the decorator that populates a forward's config-derived defaults.

    5.0 exposed ``check_model_inputs``; 5.1 split its work out into
    ``merge_with_config_defaults`` and kept the old name as a deprecated alias.
    Calling this wrapper yields a usable decorator on either.
    """
    from transformers.utils import generic

    impl = getattr(generic, "merge_with_config_defaults", None)
    if impl is not None:
        return impl
    return generic.check_model_inputs()


def default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None, **kwargs):
    """Inverse frequencies for plain, unscaled RoPE.

    Reproduces the 4.57.x ``"default"`` initializer, which 5.x removed from
    ``ROPE_INIT_FUNCTIONS``. 5.6 introduced ``"proportional"`` as the equivalent,
    but nothing fills the gap on 5.0-5.5.
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
    """Resolve a RoPE initializer without mutating Transformers' global registry."""
    if rope_type in ROPE_INIT_FUNCTIONS:
        return ROPE_INIT_FUNCTIONS[rope_type]
    if rope_type == "default":
        return ROPE_INIT_FUNCTIONS.get("proportional", default_rope_parameters)
    raise KeyError(f"unknown rope_type {rope_type!r}")


def pad_token_id_of(config):
    """Return ``config.pad_token_id``, or ``None`` where the base config omits it.

    4.57.x's ``PretrainedConfig`` always defined this, defaulting to ``None``;
    5.x removed it. The value reaches ``nn.Embedding`` as ``padding_idx``, so
    preserving ``None`` rather than substituting a codec id keeps the embedding
    tables identical across versions.
    """
    return getattr(config, "pad_token_id", None)


@lru_cache(maxsize=4)
def _mask_signature(fn) -> tuple[str, bool]:
    """Return the embeddings parameter name, and whether ``cache_position`` is accepted.

    These moved independently: 5.0 spelled it ``input_embeds`` and 5.2 renamed it
    to ``inputs_embeds``, while ``cache_position`` survived until 5.9. So 5.2-5.8
    want the new name *and* a cache position, and neither axis can be inferred
    from the other.
    """
    params = inspect.signature(fn).parameters
    embeds = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
    return embeds, "cache_position" in params


def _mask_kwargs(fn, kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    embeds = kwargs.pop("inputs_embeds", None)
    cache_position = kwargs.pop("cache_position", None)
    embeds_name, takes_cache_position = _mask_signature(fn)
    kwargs[embeds_name] = embeds
    if takes_cache_position:
        kwargs["cache_position"] = cache_position
    return kwargs


def create_causal_mask(**kwargs):
    """Version-agnostic :func:`transformers.masking_utils.create_causal_mask`."""
    return _create_causal_mask(**_mask_kwargs(_create_causal_mask, kwargs))


def create_sliding_window_causal_mask(**kwargs):
    """Version-agnostic :func:`transformers.masking_utils.create_sliding_window_causal_mask`."""
    return _create_sliding_window_causal_mask(**_mask_kwargs(_create_sliding_window_causal_mask, kwargs))
