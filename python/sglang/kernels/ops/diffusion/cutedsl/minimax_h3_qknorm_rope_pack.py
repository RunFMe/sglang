"""CuTe DSL MiniMax-H3 QK-norm + RoPE + Ulysses packing kernel."""

from __future__ import annotations

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute

from sglang.kernels.ops.diffusion.cutedsl.utils import to_fake_cute_args

HEAD_DIM = 128
ROPE_DIM = 96
WARP_SIZE = 32
WARPS_PER_BLOCK = 8
THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE
VALUES_PER_LANE = HEAD_DIM // WARP_SIZE
ROTARY_LANES = ROPE_DIM // VALUES_PER_LANE
HALF_ROTARY_LANES = ROTARY_LANES // 2
ROTARY_MASK = (1 << ROTARY_LANES) - 1


@cute.jit
def _warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    """Match the XOR-tree used by the existing CUDA QK-norm kernel."""
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def _kernel(
    tiled_copy: cute.TiledCopy,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    cos_sin_cache: cute.Tensor,
    positions: cute.Tensor,
    output: cute.Tensor,
    eps: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    lane = tidx & (WARP_SIZE - 1)
    warp = tidx >> 5

    rows = q.shape[0]
    heads = q.shape[1]
    world_size = output.shape[0]
    local_heads = heads // world_size
    total_work = rows * heads
    work = block * WARPS_PER_BLOCK + warp
    worker_stride = grid * WARPS_PER_BLOCK

    while work < total_work:
        row = work // heads
        head = work - row * heads
        destination = head // local_heads
        local_head = head - destination * local_heads
        dim_base = lane * VALUES_PER_LANE

        thread_copy = tiled_copy.get_slice(lane)
        q_global = thread_copy.partition_S(q[row, head, None])
        k_global = thread_copy.partition_S(k[row, head, None])
        v_global = thread_copy.partition_S(v[row, head, None])
        q_register = cute.make_fragment_like(q_global, q.element_type)
        k_register = cute.make_fragment_like(k_global, k.element_type)
        v_register = cute.make_fragment_like(v_global, v.element_type)
        cute.autovec_copy(q_global, q_register)
        cute.autovec_copy(k_global, k_register)
        cute.autovec_copy(v_global, v_register)

        q_values = [cutlass.Float32(0.0) for _ in range(VALUES_PER_LANE)]
        k_values = [cutlass.Float32(0.0) for _ in range(VALUES_PER_LANE)]
        v_values = [cutlass.BFloat16(0.0) for _ in range(VALUES_PER_LANE)]
        q_sum_sq = cutlass.Float32(0.0)
        k_sum_sq = cutlass.Float32(0.0)

        for item in cutlass.range_constexpr(VALUES_PER_LANE):
            dim = dim_base + item
            q_value = cutlass.Float32(q_register[item])
            k_value = cutlass.Float32(k_register[item])
            q_values[item] = q_value
            k_values[item] = k_value
            v_values[item] = v_register[item]
            q_sum_sq += q_value * q_value
            k_sum_sq += k_value * k_value

        q_scale = cute.rsqrt(_warp_sum(q_sum_sq) / HEAD_DIM + eps)
        k_scale = cute.rsqrt(_warp_sum(k_sum_sq) / HEAD_DIM + eps)

        # H3 rounds normalized Q/K to BF16 before RoPE.
        q_rounded = [cutlass.BFloat16(0.0) for _ in range(VALUES_PER_LANE)]
        k_rounded = [cutlass.BFloat16(0.0) for _ in range(VALUES_PER_LANE)]
        for item in cutlass.range_constexpr(VALUES_PER_LANE):
            dim = dim_base + item
            q_rounded[item] = cutlass.BFloat16(
                q_values[item] * q_scale * cutlass.Float32(q_weight[dim])
            )
            k_rounded[item] = cutlass.BFloat16(
                k_values[item] * k_scale * cutlass.Float32(k_weight[dim])
            )

        if lane < ROTARY_LANES:
            partner_lane = (
                lane + HALF_ROTARY_LANES
                if lane < HALF_ROTARY_LANES
                else lane - HALF_ROTARY_LANES
            )
            cache_row = cutlass.Int64(positions[row])
            half_index_base = (lane % HALF_ROTARY_LANES) * VALUES_PER_LANE
            for item in cutlass.range_constexpr(VALUES_PER_LANE):
                q_value = cutlass.Float32(q_rounded[item])
                k_value = cutlass.Float32(k_rounded[item])
                q_partner = cute.arch.shuffle_sync(
                    q_value, partner_lane, mask=ROTARY_MASK
                )
                k_partner = cute.arch.shuffle_sync(
                    k_value, partner_lane, mask=ROTARY_MASK
                )
                cos = cos_sin_cache[cache_row, half_index_base + item]
                sin = cos_sin_cache[cache_row, ROPE_DIM // 2 + half_index_base + item]
                # Match __nv_bfloat16 arithmetic: round each product before
                # the final add/subtract.
                q_main = cutlass.BFloat16(q_value * cutlass.Float32(cos))
                q_cross = cutlass.BFloat16(q_partner * cutlass.Float32(sin))
                k_main = cutlass.BFloat16(k_value * cutlass.Float32(cos))
                k_cross = cutlass.BFloat16(k_partner * cutlass.Float32(sin))
                if lane < HALF_ROTARY_LANES:
                    q_value = cutlass.Float32(q_main) - cutlass.Float32(q_cross)
                    k_value = cutlass.Float32(k_main) - cutlass.Float32(k_cross)
                else:
                    q_value = cutlass.Float32(q_main) + cutlass.Float32(q_cross)
                    k_value = cutlass.Float32(k_main) + cutlass.Float32(k_cross)
                q_rounded[item] = cutlass.BFloat16(q_value)
                k_rounded[item] = cutlass.BFloat16(k_value)

        q_output_global = thread_copy.partition_D(
            output[destination, row, local_head, 0, None]
        )
        k_output_global = thread_copy.partition_D(
            output[destination, row, local_head, 1, None]
        )
        v_output_global = thread_copy.partition_D(
            output[destination, row, local_head, 2, None]
        )
        q_output_register = cute.make_fragment_like(
            q_output_global, output.element_type
        )
        k_output_register = cute.make_fragment_like(
            k_output_global, output.element_type
        )
        v_output_register = cute.make_fragment_like(
            v_output_global, output.element_type
        )
        for item in cutlass.range_constexpr(VALUES_PER_LANE):
            q_output_register[item] = q_rounded[item]
            k_output_register[item] = k_rounded[item]
            v_output_register[item] = v_values[item]
        cute.autovec_copy(q_output_register, q_output_global)
        cute.autovec_copy(k_output_register, k_output_global)
        cute.autovec_copy(v_output_register, v_output_global)

        work += worker_stride


class _FusedQKNormRopePack:
    def __init__(self, num_sms: int, blocks_per_sm: int):
        self.grid = num_sms * blocks_per_sm

    @cute.jit
    def __call__(
        self,
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        output,
        eps: cutlass.Float32 = cutlass.Float32(1e-5),  # noqa: B008
        stream: cuda.CUstream = cuda.CUstream(  # noqa: B008
            cuda.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            q.element_type,
            num_bits_per_copy=64,
        )
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout(WARP_SIZE),
            cute.make_layout(VALUES_PER_LANE),
        )
        _kernel(
            tiled_copy,
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            output,
            eps,
        ).launch(
            grid=[self.grid, 1, 1],
            block=[THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )


def _fake_argument(tensor: torch.Tensor):
    if tensor.dtype == torch.int32:
        dtype = cutlass.Int32
    elif tensor.dtype == torch.int64:
        dtype = cutlass.Int64
    else:
        return to_fake_cute_args(tensor)
    return cute.runtime.make_fake_tensor(
        dtype,
        (cute.sym_int(),),
        (1,),
        memspace=cute.AddressSpace.gmem,
        assumed_align=16,
    )


@cache
def _compile(
    position_dtype: torch.dtype,
    device_index: int,
    num_sms: int,
    blocks_per_sm: int,
):
    # Runtime dimensions and strides remain symbolic; only the H3 signature
    # and launch geometry specialize the generated kernel.
    q = torch.empty_strided(
        (1, 1, HEAD_DIM),
        (3 * HEAD_DIM, HEAD_DIM, 1),
        dtype=torch.bfloat16,
        device=torch.device("cuda", device_index),
    )
    k = torch.empty_strided(
        (1, 1, HEAD_DIM),
        (3 * HEAD_DIM, HEAD_DIM, 1),
        dtype=torch.bfloat16,
        device=q.device,
    )
    v = torch.empty_strided(
        (1, 1, HEAD_DIM),
        (3 * HEAD_DIM, HEAD_DIM, 1),
        dtype=torch.bfloat16,
        device=q.device,
    )
    weight = torch.empty(HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    cache = torch.empty(1, ROPE_DIM, dtype=torch.bfloat16, device=q.device)
    positions = torch.empty(1, dtype=position_dtype, device=q.device)
    output = torch.empty((1, 1, 1, 3, HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    args = (q, k, v, weight, weight, cache, positions, output)
    return cute.compile(
        _FusedQKNormRopePack(num_sms, blocks_per_sm),
        *(_fake_argument(tensor) for tensor in args),
        options="--enable-tvm-ffi",
    )


def minimax_h3_qknorm_rope_pack_cute(
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
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    # Four persistent CTAs per SM won the H200 production-shape sweep.
    blocks_per_sm = 4
    compiled = _compile(
        positions.dtype,
        device_index,
        properties.multi_processor_count,
        blocks_per_sm,
    )
    output = out.view(world_size, q.shape[0], q.shape[1] // world_size, 3, HEAD_DIM)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        output,
        eps,
        stream,
    )
    return out


__all__ = ["minimax_h3_qknorm_rope_pack_cute"]
