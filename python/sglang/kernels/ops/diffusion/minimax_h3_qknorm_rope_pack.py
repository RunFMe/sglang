"""Fused MiniMax-H3 QK-norm + RoPE + destination-major QKV packing."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
)

if TYPE_CHECKING:
    import torch

HEAD_DIM = 128
ROPE_DIM = 96
_CUDA_SM80_PLUS = frozenset({CapabilityRequirement.cuda(min_sm=(8, 0))})


@lru_cache(maxsize=1)
def _has_cute_dsl() -> bool:
    try:
        return importlib.util.find_spec("cutlass.cute") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _supports_h3_signature(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None,
    *,
    require_cuda: bool = True,
) -> bool:
    import torch

    if (
        (require_cuda and not q.is_cuda)
        or q.ndim != 3
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[-1] != HEAD_DIM
        or q.dtype != torch.bfloat16
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.device != k.device
        or q.device != v.device
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or v.stride(-1) != 1
        or world_size <= 1
        or q.shape[1] % world_size
    ):
        return False
    if (
        q_weight.shape != (HEAD_DIM,)
        or k_weight.shape != (HEAD_DIM,)
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
        or q_weight.device != q.device
        or k_weight.device != q.device
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    ):
        return False
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != ROPE_DIM
        or cos_sin_cache.dtype != torch.bfloat16
        or cos_sin_cache.device != q.device
        or not cos_sin_cache.is_contiguous()
        or positions.shape != (q.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != q.device
        or not positions.is_contiguous()
    ):
        return False
    if out is None:
        return True
    expected = (
        world_size,
        q.shape[0],
        q.shape[1] // world_size,
        3 * HEAD_DIM,
    )
    return (
        out.shape == expected
        and out.dtype == q.dtype
        and out.device == q.device
        and out.is_contiguous()
    )


def _validate_h3_signature(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None,
    *,
    require_cuda: bool = True,
) -> None:
    if not _supports_h3_signature(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size,
        out,
        require_cuda=require_cuda,
    ):
        raise ValueError(
            "MiniMax-H3 fused QK-norm/RoPE/pack requires CUDA BF16 Q/K/V "
            "[rows, heads, 128], BF16 weights [128], a contiguous BF16 "
            "cos|sin cache [positions, 96], contiguous int32/int64 positions, "
            "and a Ulysses world size greater than one that divides heads"
        )


def _output(q: torch.Tensor, world_size: int, out: torch.Tensor | None):
    import torch

    if out is not None:
        return out
    return torch.empty(
        (
            world_size,
            q.shape[0],
            q.shape[1] // world_size,
            3 * HEAD_DIM,
        ),
        dtype=q.dtype,
        device=q.device,
    )


class MiniMaxH3QKNormRopePackOp(BaseFusedOp):
    """H3 post-projection transform writing directly to the Ulysses send buffer.

    Optimized backends do not modify Q/K. The CUDA compatibility path uses the
    existing in-place QK-norm/RoPE kernel, so callers must treat Q/K as consumed.
    """

    op = "diffusion.minimax_h3_qknorm_rope_pack"
    priority = (
        KernelBackend.CUTE_DSL,
        KernelBackend.TRITON,
        KernelBackend.TORCH,
    )
    capabilities: ClassVar[dict] = {
        KernelBackend.CUTE_DSL: _CUDA_SM80_PLUS,
        KernelBackend.TRITON: _CUDA_SM80_PLUS,
    }
    format_signature = FormatSignature(
        supported_dtypes=("bfloat16",),
        in_place=True,
        description=(
            "MiniMax-H3 BF16 QK RMSNorm + partial NeoX RoPE + destination-major "
            "Ulysses QKV packing"
        ),
    )
    descriptions: ClassVar[dict] = {
        KernelBackend.CUTE_DSL: (
            "MiniMax-H3 QK-norm + RoPE + Ulysses packing (CuTe DSL)."
        ),
        KernelBackend.TRITON: ("MiniMax-H3 QK-norm + RoPE + Ulysses packing (Triton)."),
        KernelBackend.TORCH: (
            "MiniMax-H3 QK-norm + RoPE + Ulysses packing (Torch reference)."
        ),
    }

    def backend_eligible(
        self,
        backend: KernelBackend,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        world_size: int,
        out: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> bool:
        import torch

        if not super().backend_eligible(backend):
            return False
        if not _supports_h3_signature(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            out,
        ):
            return False
        # The checked-in launch geometry is tuned on H100/H200. Untested
        # architectures retain the existing CUDA implementation until their
        # profile is measured; the Triton backend remains force-selectable by
        # the benchmark harness.
        is_sm90 = torch.cuda.get_device_capability(q.device) == (9, 0)
        if backend is KernelBackend.CUTE_DSL:
            return is_sm90 and _has_cute_dsl()
        if backend is KernelBackend.TRITON:
            return is_sm90 and not _has_cute_dsl()
        return False

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        world_size: int,
        out: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        import torch

        _validate_h3_signature(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            out,
            require_cuda=False,
        )

        def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_float = x.float()
            scale = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
            return (x_float * scale * weight.float()).to(torch.bfloat16)

        def rope(x: torch.Tensor) -> torch.Tensor:
            cache = cos_sin_cache.index_select(0, positions.long())[:, None, :]
            cos, sin = cache.split(ROPE_DIM // 2, dim=-1)
            first, second, passthrough = x.split(
                (ROPE_DIM // 2, ROPE_DIM // 2, HEAD_DIM - ROPE_DIM), dim=-1
            )
            first_main = (first * cos).to(torch.bfloat16)
            first_cross = (second * sin).to(torch.bfloat16)
            second_main = (second * cos).to(torch.bfloat16)
            second_cross = (first * sin).to(torch.bfloat16)
            return torch.cat(
                (
                    (first_main - first_cross).to(torch.bfloat16),
                    (second_main + second_cross).to(torch.bfloat16),
                    passthrough,
                ),
                dim=-1,
            )

        q_norm = rope(norm(q, q_weight))
        k_norm = rope(norm(k, k_weight))
        output = _output(q, world_size, out)
        output_view = output.view(
            world_size,
            q.shape[0],
            q.shape[1] // world_size,
            3,
            HEAD_DIM,
        )
        for index, tensor in enumerate((q_norm, k_norm, v)):
            output_view[..., index, :].copy_(
                tensor.view(
                    q.shape[0], world_size, q.shape[1] // world_size, HEAD_DIM
                ).permute(1, 0, 2, 3)
            )
        q.copy_(q_norm)
        k.copy_(k_norm)
        return output

    def forward_cute_dsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        world_size: int,
        out: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion.cutedsl.minimax_h3_qknorm_rope_pack import (
            minimax_h3_qknorm_rope_pack_cute,
        )

        _validate_h3_signature(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            out,
        )
        return minimax_h3_qknorm_rope_pack_cute(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            _output(q, world_size, out),
            eps,
        )

    def forward_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        world_size: int,
        out: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion.triton.minimax_h3_qknorm_rope_pack import (
            minimax_h3_qknorm_rope_pack_triton,
        )

        _validate_h3_signature(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            out,
        )
        return minimax_h3_qknorm_rope_pack_triton(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            _output(q, world_size, out),
            eps,
        )

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        world_size: int,
        out: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        from sglang.kernels.ops.diffusion.qknorm_rope import (
            fused_inplace_qknorm_rope,
        )
        from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
            pack_qkv_destination_major,
        )

        _validate_h3_signature(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            world_size,
            out,
        )
        fused_inplace_qknorm_rope(
            q,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            is_neox=True,
            eps=eps,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
            round_norm_before_rope=True,
        )
        return pack_qkv_destination_major(
            q, k, v, world_size, out=_output(q, world_size, out)
        )


_MINIMAX_H3_QKNORM_ROPE_PACK = register_fused_op(
    MiniMaxH3QKNormRopePackOp(),
    __name__,
    "_MINIMAX_H3_QKNORM_ROPE_PACK",
)


def minimax_h3_qknorm_rope_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None = None,
    eps: float = 1e-5,
    *,
    backend: KernelBackend | None = None,
) -> torch.Tensor:
    return _MINIMAX_H3_QKNORM_ROPE_PACK(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size,
        out,
        eps,
        backend=backend,
    )


__all__ = [
    "MiniMaxH3QKNormRopePackOp",
    "minimax_h3_qknorm_rope_pack",
]
