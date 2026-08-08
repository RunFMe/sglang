"""Triton MiniMax-H3 QK-norm + RoPE + Ulysses packing fallback."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
ROPE_DIM = 96


@triton.jit
def _kernel(
    output_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    total_work,
    rows,
    heads,
    local_heads,
    stride_q_row,
    stride_q_head,
    stride_k_row,
    stride_k_head,
    stride_v_row,
    stride_v_head,
    stride_cache_row,
    eps: tl.constexpr,
    HEAD_D: tl.constexpr,
    ROPE_D: tl.constexpr,
):
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    lanes = tl.arange(0, 32)

    for work in tl.range(program, total_work, programs, num_stages=1):
        row = work // heads
        head = work - row * heads
        q_base = q_ptr + row * stride_q_row + head * stride_q_head
        k_base = k_ptr + row * stride_k_row + head * stride_k_head
        v_base = v_ptr + row * stride_v_row + head * stride_v_head

        # Match the existing CUDA kernel: each lane accumulates four adjacent
        # dimensions serially, then the warp performs its reduction.
        dim0 = lanes * 4
        q0 = tl.load(q_base + dim0).to(tl.float32)
        q1 = tl.load(q_base + dim0 + 1).to(tl.float32)
        q2 = tl.load(q_base + dim0 + 2).to(tl.float32)
        q3 = tl.load(q_base + dim0 + 3).to(tl.float32)
        k0 = tl.load(k_base + dim0).to(tl.float32)
        k1 = tl.load(k_base + dim0 + 1).to(tl.float32)
        k2 = tl.load(k_base + dim0 + 2).to(tl.float32)
        k3 = tl.load(k_base + dim0 + 3).to(tl.float32)
        q_partial = q0 * q0
        q_partial += q1 * q1
        q_partial += q2 * q2
        q_partial += q3 * q3
        k_partial = k0 * k0
        k_partial += k1 * k1
        k_partial += k2 * k2
        k_partial += k3 * k3
        q_scale = tl.rsqrt(tl.sum(q_partial, axis=0) / HEAD_D + eps)
        k_scale = tl.rsqrt(tl.sum(k_partial, axis=0) / HEAD_D + eps)

        destination = head // local_heads
        local_head = head - destination * local_heads
        output_base = ((destination * rows + row) * local_heads + local_head) * (
            3 * HEAD_D
        )
        position = tl.load(positions_ptr + row).to(tl.int64)

        for item in tl.static_range(0, 4):
            dim = dim0 + item
            q_value = tl.load(q_base + dim).to(tl.float32)
            k_value = tl.load(k_base + dim).to(tl.float32)
            v_value = tl.load(v_base + dim)
            q_weight = tl.load(q_weight_ptr + dim).to(tl.float32)
            k_weight = tl.load(k_weight_ptr + dim).to(tl.float32)
            q_norm = (q_value * q_scale * q_weight).to(tl.bfloat16)
            k_norm = (k_value * k_scale * k_weight).to(tl.bfloat16)

            rotary_mask = dim < ROPE_D
            half_dim = dim % (ROPE_D // 2)
            partner_dim = tl.where(
                dim < ROPE_D // 2,
                dim + ROPE_D // 2,
                dim - ROPE_D // 2,
            )
            q_partner = tl.load(q_base + partner_dim, mask=rotary_mask, other=0.0).to(
                tl.float32
            )
            k_partner = tl.load(k_base + partner_dim, mask=rotary_mask, other=0.0).to(
                tl.float32
            )
            q_partner_weight = tl.load(
                q_weight_ptr + partner_dim, mask=rotary_mask, other=0.0
            ).to(tl.float32)
            k_partner_weight = tl.load(
                k_weight_ptr + partner_dim, mask=rotary_mask, other=0.0
            ).to(tl.float32)
            q_partner = (q_partner * q_scale * q_partner_weight).to(tl.bfloat16)
            k_partner = (k_partner * k_scale * k_partner_weight).to(tl.bfloat16)
            cos = tl.load(
                cos_sin_cache_ptr + position * stride_cache_row + half_dim,
                mask=rotary_mask,
                other=0.0,
            )
            sin = tl.load(
                cos_sin_cache_ptr
                + position * stride_cache_row
                + ROPE_D // 2
                + half_dim,
                mask=rotary_mask,
                other=0.0,
            )
            q_main = (q_norm.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
            q_cross = (q_partner.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
            k_main = (k_norm.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
            k_cross = (k_partner.to(tl.float32) * sin.to(tl.float32)).to(tl.bfloat16)
            sign = tl.where(dim < ROPE_D // 2, -1.0, 1.0)
            q_rotated = (q_main.to(tl.float32) + sign * q_cross.to(tl.float32)).to(
                tl.bfloat16
            )
            k_rotated = (k_main.to(tl.float32) + sign * k_cross.to(tl.float32)).to(
                tl.bfloat16
            )
            q_out = tl.where(rotary_mask, q_rotated, q_norm)
            k_out = tl.where(rotary_mask, k_rotated, k_norm)

            tl.store(output_ptr + output_base + dim, q_out)
            tl.store(output_ptr + output_base + HEAD_D + dim, k_out)
            tl.store(output_ptr + output_base + 2 * HEAD_D + dim, v_value)


def minimax_h3_qknorm_rope_pack_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    world_size: int,
    out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rows, heads, _ = q.shape
    total_work = rows * heads
    if total_work == 0:
        return out
    local_heads = heads // world_size
    properties = torch.cuda.get_device_properties(q.device)
    grid = (properties.multi_processor_count * 8,)
    _kernel[grid](
        out,
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        total_work,
        rows,
        heads,
        local_heads,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        cos_sin_cache.stride(0),
        eps=eps,
        HEAD_D=HEAD_DIM,
        ROPE_D=ROPE_DIM,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["minimax_h3_qknorm_rope_pack_triton"]
