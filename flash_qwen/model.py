"""
Qwen3.5-35B-A3B model with flash-loaded MoE experts.

Uses the correct qwen3_5.py architecture from mlx-lm 0.31.1+, with only the
MoE block replaced to stream expert weights from SSD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import the correct model components from mlx-lm
from mlx_lm.models.qwen3_5 import (
    Attention,
    GatedDeltaNet,
    MLP,
    RMSNormGated,
    SparseMoeBlock,
    TextModelArgs,
)
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.cache import ArraysCache, KVCache

from flash_qwen._native import FlashExpertManager
from flash_qwen.turbo_quant import TurboQuantCache


class FlashSparseMoeBlock(nn.Module):
    """MoE block that streams expert weights from SSD via the Rust loader."""

    def __init__(self, args: TextModelArgs, layer_idx: int, expert_manager: FlashExpertManager):
        super().__init__()
        dim = args.hidden_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        # Router stays in RAM
        self.gate = nn.Linear(dim, self.num_experts, bias=False)

        # Shared expert stays in RAM
        self.shared_expert = MLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

        # No switch_mlp -- expert weights live on SSD
        self.expert_manager = expert_manager
        self.layer_idx = layer_idx

    def __call__(self, x: mx.array) -> mx.array:
        # 1. Router
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # 2. Force evaluation to get concrete expert indices
        mx.eval(inds)

        orig_shape = inds.shape
        flat_inds = inds.reshape(-1).tolist()
        unique_experts = sorted(set(flat_inds))

        # 3. Load experts from SSD/cache
        expert_data = self.expert_manager.load_experts(self.layer_idx, unique_experts)

        # 4. Remap indices
        remap = {orig: local for local, orig in enumerate(unique_experts)}
        flat_remapped = [remap[i] for i in flat_inds]
        local_inds = mx.array(flat_remapped, dtype=mx.uint32).reshape(orig_shape)

        # 5. Build weight tensors
        bits = expert_data["quant_bits"]
        group_size = expert_data["quant_group_size"]

        def make_weight(name):
            data = expert_data[name]
            shape = expert_data[f"{name}_shape"]
            return mx.array(np.frombuffer(data, dtype=np.uint32).reshape(shape))

        def make_bf16(name):
            data = expert_data[name]
            shape = expert_data[f"{name}_shape"]
            return mx.array(np.frombuffer(data, dtype=np.uint16).reshape(shape)).view(mx.bfloat16)

        gate_w = make_weight("gate_weight")
        gate_s = make_bf16("gate_scales")
        gate_b = make_bf16("gate_biases")
        up_w = make_weight("up_weight")
        up_s = make_bf16("up_scales")
        up_b = make_bf16("up_biases")
        down_w = make_weight("down_weight")
        down_s = make_bf16("down_scales")
        down_b = make_bf16("down_biases")

        # 6. Compute using gather_qmm (following SwitchGLU pattern)
        x_exp = mx.expand_dims(x, (-2, -3))

        do_sort = local_inds.size >= 64
        idx = local_inds
        inv_order = None
        if do_sort:
            M = local_inds.shape[-1]
            flat_idx = local_inds.flatten()
            order = mx.argsort(flat_idx)
            inv_order = mx.argsort(order)
            x_exp = x_exp.flatten(0, -3)[order // M]
            idx = flat_idx[order]

        qmm_kwargs = dict(transpose=True, group_size=group_size, bits=bits, sorted_indices=do_sort)

        x_gate = mx.gather_qmm(x_exp, gate_w, gate_s, gate_b, rhs_indices=idx, **qmm_kwargs)
        x_up = mx.gather_qmm(x_exp, up_w, up_s, up_b, rhs_indices=idx, **qmm_kwargs)
        x_act = nn.silu(x_gate) * x_up
        x_down = mx.gather_qmm(x_act, down_w, down_s, down_b, rhs_indices=idx, **qmm_kwargs)

        if do_sort:
            x_down = x_down[inv_order]
            x_down = mx.unflatten(x_down, 0, local_inds.shape)

        x_down = x_down.squeeze(-2)
        y = (x_down * scores[..., None]).sum(axis=-2)

        # 7. Shared expert
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


class FlashDecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int, expert_manager: FlashExpertManager):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = FlashSparseMoeBlock(args, layer_idx, expert_manager)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(self, x, mask=None, cache=None):
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


class FlashTextModel(nn.Module):
    def __init__(self, args: TextModelArgs, expert_manager: FlashExpertManager):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            FlashDecoderLayer(args=args, layer_idx=i, expert_manager=expert_manager)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(self, inputs, cache=None):
        hidden_states = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c)

        return self.norm(hidden_states)


class FlashLanguageModel(nn.Module):
    """Matches the TextModel structure from qwen3_5.py."""

    def __init__(self, args: TextModelArgs, expert_manager: FlashExpertManager):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlashTextModel(args, expert_manager)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, kv_bits=None):
        def _make_kv():
            if kv_bits is not None:
                return TurboQuantCache(
                    head_dim=self.args.head_dim, bits=kv_bits,
                )
            return KVCache()

        return [ArraysCache(size=2) if l.is_linear else _make_kv() for l in self.layers]


class Model(nn.Module):
    """Top-level model matching qwen3_5.Model's weight structure.

    Weights live under language_model.model.layers.N...
    """

    def __init__(self, args: TextModelArgs, expert_manager: FlashExpertManager):
        super().__init__()
        self.args = args
        self.language_model = FlashLanguageModel(args, expert_manager)

    def __call__(self, inputs, cache=None):
        return self.language_model(inputs, cache=cache)

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self, kv_bits=None):
        return self.language_model.make_cache(kv_bits=kv_bits)
