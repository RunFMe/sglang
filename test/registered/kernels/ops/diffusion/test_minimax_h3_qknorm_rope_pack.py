"""Bitwise tests for the MiniMax-H3 fused QK-norm/RoPE/Ulysses pack op."""

import sys

import pytest
import torch
from sglang.kernels.ops.diffusion.minimax_h3_qknorm_rope_pack import (
    MiniMaxH3QKNormRopePackOp,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
HEAD_DIM = 128
ROPE_DIM = 96


def _inputs(rows: int, heads: int, position_dtype: torch.dtype):
    generator = torch.Generator(device=DEVICE).manual_seed(rows * 97 + heads)
    qkv = torch.randn(
        rows,
        3 * heads * HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    q, k, v = (
        tensor.view(rows, heads, HEAD_DIM)
        for tensor in qkv.split(heads * HEAD_DIM, dim=-1)
    )
    q_weight = torch.randn(
        HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    k_weight = torch.randn(
        HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    angles = torch.randn(
        rows + 17,
        ROPE_DIM // 2,
        device=DEVICE,
        dtype=torch.float32,
        generator=generator,
    )
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.randperm(
        rows + 17, device=DEVICE, generator=generator, dtype=torch.int64
    )[:rows].to(position_dtype)
    return q, k, v, q_weight, k_weight, cache, positions


@pytest.mark.parametrize(
    "rows,heads,world_size",
    [
        (1, 56, 4),
        (65, 56, 8),
        (257, 28, 4),
        (1024, 14, 2),
    ],
)
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("backend", ["triton", "cute_dsl"])
def test_fused_qknorm_rope_pack_is_bit_exact(
    rows: int,
    heads: int,
    world_size: int,
    position_dtype: torch.dtype,
    backend: str,
):
    if backend == "cute_dsl" and torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("the checked-in CuTe launch profile is enabled on SM90")

    q, k, v, q_weight, k_weight, cache, positions = _inputs(rows, heads, position_dtype)
    op = MiniMaxH3QKNormRopePackOp()
    q_ref, k_ref, v_ref = q.clone(), k.clone(), v.clone()
    expected = op.forward_cuda(
        q_ref,
        k_ref,
        v_ref,
        q_weight,
        k_weight,
        cache,
        positions,
        world_size,
    )
    out = torch.empty_like(expected)
    q_actual, k_actual, v_actual = q.clone(), k.clone(), v.clone()
    implementation = op.forward_triton if backend == "triton" else op.forward_cute_dsl
    actual = implementation(
        q_actual,
        k_actual,
        v_actual,
        q_weight,
        k_weight,
        cache,
        positions,
        world_size,
        out=out,
    )

    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)
    # Optimized backends only consume the projection views and never need to
    # write normalized Q/K back to the large fused-QKV buffer.
    assert torch.equal(q_actual, q)
    assert torch.equal(k_actual, k)
    assert torch.equal(v_actual, v)


def test_fused_qknorm_rope_pack_rejects_non_h3_signature():
    q = torch.randn(4, 8, 64, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(64, device=DEVICE, dtype=torch.bfloat16)
    cache = torch.randn(4, 64, device=DEVICE, dtype=torch.bfloat16)
    positions = torch.arange(4, device=DEVICE)
    with pytest.raises(ValueError, match="requires CUDA BF16 Q/K/V"):
        MiniMaxH3QKNormRopePackOp().forward_triton(
            q, q, q, weight, weight, cache, positions, 2
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
