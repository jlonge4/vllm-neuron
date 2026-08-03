# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for MiniMax-M2 BF16 model.

MiniMax-M2/M2.5 checkpoint format:
- FP8 (float8_e4m3fn) weights with block-wise scales (weight_block_size=[128,128])
- Each weight tensor has a corresponding weight_scale_inv tensor
- Expert naming: w1=gate_proj, w2=down_proj, w3=up_proj
- Router: block_sparse_moe.gate.weight + e_score_correction_bias
- Attention: q/k/v/o_proj with q_norm/k_norm
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


BLOCK_SIZE = 128


def _dequant_fp8_block(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor
) -> torch.Tensor:
    """Dequantize block-wise FP8 weight to BF16.

    Args:
        weight_fp8: [M, N] in float8_e4m3fn
        scale_inv: [M/block, N/block] inverse scales

    Returns:
        [M, N] in bfloat16
    """
    M, N = weight_fp8.shape
    bm = BLOCK_SIZE
    bn = BLOCK_SIZE
    # Reshape weight into blocks
    weight_f32 = weight_fp8.to(torch.float32).reshape(M // bm, bm, N // bn, bn)
    # scale_inv is [M/bm, N/bn] — broadcast over block dims
    scale = scale_inv.float().reshape(M // bm, 1, N // bn, 1)
    dequant = (weight_f32 * scale).reshape(M, N)
    return dequant.to(torch.bfloat16)


def fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    num_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    num_kv_replicas: int = 1,
    kv_num_shards: int | None = None,
) -> SafetensorsWeightLoader:
    """Fuses Q, K, V weights with per-tensor sharding. Dequants FP8 to BF16.

    Receives 6 slices: [Q_weight, Q_scale, K_weight, K_scale, V_weight, V_scale].
    """
    kv_shards = kv_num_shards if kv_num_shards is not None else num_shards

    if kv_shards >= num_kv_heads:
        num_kv_heads_per_rank = 1
    else:
        num_kv_heads_per_rank = num_kv_heads // kv_shards

    kv_size_per_rank = num_kv_heads_per_rank * head_dim

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 6, (
            f"fused_qkv_weight_loader expects [Q_w, Q_s, K_w, K_s, V_w, V_s], got {len(slices)}"
        )

        q_w, q_s, k_w, k_s, v_w, v_s = slices

        # Dequant each projection
        q_full = _dequant_fp8_block(q_w[:], q_s[:])
        k_full = _dequant_fp8_block(k_w[:], k_s[:])
        v_full = _dequant_fp8_block(v_w[:], v_s[:])

        # Shard Q
        q_start = rank * q_size
        q_tensor = q_full[q_start : q_start + q_size, :]

        # Shard K/V
        if kv_shards != num_shards:
            tp_rank_for_kv = rank // (num_shards // kv_shards)
            kv_replicas_for_kv = max(kv_shards // num_kv_heads, 1)
            kv_rank = tp_rank_for_kv // kv_replicas_for_kv
        else:
            kv_rank = rank // num_kv_replicas

        if kv_shards >= num_kv_heads:
            if kv_rank < num_kv_heads:
                kv_start = kv_rank * head_dim
                k_tensor = k_full[kv_start : kv_start + head_dim, :]
                v_tensor = v_full[kv_start : kv_start + head_dim, :]
            else:
                k_tensor = torch.zeros(head_dim, hidden_size, dtype=torch.bfloat16)
                v_tensor = torch.zeros(head_dim, hidden_size, dtype=torch.bfloat16)
        else:
            kv_start = kv_rank * kv_size_per_rank
            k_tensor = k_full[kv_start : kv_start + kv_size_per_rank, :]
            v_tensor = v_full[kv_start : kv_start + kv_size_per_rank, :]

        result = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)
        return result.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def o_proj_weight_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
) -> SafetensorsWeightLoader:
    """Weight loader for o_proj that shards input dimension. Dequants FP8.

    Receives 2 slices: [weight, scale_inv].
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 2, f"o_proj expects [weight, scale], got {len(slices)}"
        w, s = slices
        tensor = _dequant_fp8_block(w[:], s[:])

        # Checkpoint is [hidden, out], shard on dim 1
        start = rank * shard_size
        tensor = tensor[:, start : start + shard_size]

        return tensor.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def expert_gate_up_weight_sharding_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
    num_experts: int = 256,
) -> SafetensorsWeightLoader:
    """Weight loader for gate_up_proj from per-expert FP8 checkpoint tensors.

    Receives 4*num_experts slices:
    [w1_0_weight, w1_0_scale, ..., w1_{E-1}_weight, w1_{E-1}_scale,
     w3_0_weight, w3_0_scale, ..., w3_{E-1}_weight, w3_{E-1}_scale]

    w1=gate_proj, w3=up_proj. Each weight is [I, H] in FP8.
    Dequants, shards gate and up independently on I, concatenates to [E, H, I_shard*2].
    """
    i_per_rank = shard_size // 2

    def transform(slices: list, rank: int) -> torch.Tensor:
        E = num_experts
        assert len(slices) == 4 * E, (
            f"Expected {4 * E} slices (gate_w+gate_s+up_w+up_s), got {len(slices)}"
        )

        # First 2*E slices are gate (w1): weight, scale alternating per expert
        gate_weights = slices[: 2 * E]
        # Next 2*E slices are up (w3): weight, scale alternating per expert
        up_weights = slices[2 * E :]

        gates = []
        ups = []
        for i in range(E):
            gw = gate_weights[2 * i][:]
            gs = gate_weights[2 * i + 1][:]
            gates.append(_dequant_fp8_block(gw, gs))

            uw = up_weights[2 * i][:]
            us = up_weights[2 * i + 1][:]
            ups.append(_dequant_fp8_block(uw, us))

        # Stack: [E, I, H] -> transpose -> [E, H, I]
        gates_t = torch.stack(gates).transpose(1, 2)
        ups_t = torch.stack(ups).transpose(1, 2)

        # Shard gate and up independently on I dimension
        start_idx = (rank % num_shards) * i_per_rank
        gates_shard = gates_t[:, :, start_idx : start_idx + i_per_rank]
        ups_shard = ups_t[:, :, start_idx : start_idx + i_per_rank]

        return torch.cat([gates_shard, ups_shard], dim=2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def expert_down_weight_sharding_loader(
    shard_size: int,
    num_shards: int,
    hidden_size: int,
    num_experts: int = 256,
) -> SafetensorsWeightLoader:
    """Weight loader for down_proj from per-expert FP8 checkpoint tensors.

    Receives 2*num_experts slices: [w2_0_weight, w2_0_scale, ..., w2_{E-1}_weight, w2_{E-1}_scale].
    w2=down_proj. Each weight is [H, I] in FP8.
    Dequants, stacks into [E, I, H], shards on I.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        E = num_experts
        assert len(slices) == 2 * E, f"Expected {2 * E} slices, got {len(slices)}"

        downs = []
        for i in range(E):
            w = slices[2 * i][:]
            s = slices[2 * i + 1][:]
            # Dequant [H, I] -> transpose to [I, H]
            downs.append(_dequant_fp8_block(w, s).T)

        # Stack to [E, I, H]
        downs_stacked = torch.stack(downs)

        # Shard on I dimension (dim=1)
        start_idx = (rank % num_shards) * shard_size
        return downs_stacked[:, start_idx : start_idx + shard_size, :].contiguous()

    return SafetensorsWeightLoader(transform=transform)
