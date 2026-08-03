# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M2 Configuration
================================
"""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class MiniMaxM2Config:
    """
    Configuration for MiniMax-M2/M2.5 model.

    Architecture overview:
    - Transformer decoder with MoE (Mixture of Experts) feed-forward layers
    - GQA (Grouped Query Attention) with 48 Q heads and 8 KV heads
    - Partial RoPE (rotary_dim=64 of head_dim=128)
    - SiLU activation in expert feed-forward
    - Per-layer QK-norm (RMSNorm)
    - Sigmoid router with bias
    - No shared expert, no sliding window, no attention sinks
    """

    # ── Model architecture ───────────────────────────────────────────────
    vocab_size: int = 200064
    hidden_size: int = 3072
    num_hidden_layers: int = 62
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 1536  # Per-expert intermediate size
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.bfloat16

    # ── MoE configuration ────────────────────────────────────────────────
    num_experts: int = 256
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True

    # ── RoPE settings ────────────────────────────────────────────────────
    rope_theta: float = 5000000.0
    rotary_dim: int = 64  # Only first 64 of 128 head_dim gets RoPE

    # ── QK-norm ──────────────────────────────────────────────────────────
    use_qk_norm: bool = True

    # ── Sequence settings ────────────────────────────────────────────────
    max_position_embeddings: int = 196608

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Create config from HuggingFace config + NeuronConfig."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config is None
            ):
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}

        # Map HF field name num_local_experts -> num_experts
        mapped = dict(config_dict)
        if "num_local_experts" in mapped and "num_experts" not in mapped:
            mapped["num_experts"] = mapped.pop("num_local_experts")

        filtered_dict = {k: v for k, v in mapped.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
