"""Benchmark MiniMax-H3 QK-norm + RoPE + Ulysses packing backends."""

from dataclasses import dataclass

import torch
import triton
import triton.testing
from sglang.kernels.jit.benchmark.utils import (
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.kernels.ops.diffusion.minimax_h3_qknorm_rope_pack import (
    MiniMaxH3QKNormRopePackOp,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=24, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEAD_DIM = 128
ROPE_DIM = 96
DEVICE = "cuda"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    rows: int
    heads: int
    world_size: int


BENCH_CASES = (
    CaseSpec("tail_r65_h56_u4", 65, 56, 4),
    CaseSpec("short_r1024_h56_u4", 1024, 56, 4),
    CaseSpec("medium_r4096_h56_u4", 4096, 56, 4),
    CaseSpec("production_r7936_h56_u4", 7936, 56, 4),
    CaseSpec("production_r8304_h56_u4", 8304, 56, 4),
    CaseSpec("production_tp2_r8304_h28_u4", 8304, 28, 4),
    CaseSpec("production_tp4_r8304_h14_u2", 8304, 14, 2),
    CaseSpec("long_r16384_h56_u8", 16384, 56, 8),
)
CASE_BY_NAME = {case.name: case for case in BENCH_CASES}
CASE_NAMES = get_benchmark_range(
    full_range=[case.name for case in BENCH_CASES],
    ci_range=["short_r1024_h56_u4", "production_r8304_h56_u4"],
)
LINE_VALS = ["current", "triton", "cute_dsl"]
LINE_NAMES = [
    "Current fused QKNorm/RoPE + pack",
    "Fused Triton direct pack",
    "Fused CuTe DSL direct pack",
]
STYLES = [("red", "-"), ("green", "--"), ("blue", "-.")]


def make_inputs(case: CaseSpec):
    generator = torch.Generator(device=DEVICE).manual_seed(
        case.rows * 101 + case.heads * 17 + case.world_size
    )
    qkv = torch.randn(
        case.rows,
        3 * case.heads * HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    q, k, v = (
        tensor.view(case.rows, case.heads, HEAD_DIM)
        for tensor in qkv.split(case.heads * HEAD_DIM, dim=-1)
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
        case.rows,
        ROPE_DIM // 2,
        device=DEVICE,
        generator=generator,
    )
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    positions = torch.arange(case.rows, device=DEVICE, dtype=torch.int64)
    output = torch.empty(
        case.world_size,
        case.rows,
        case.heads // case.world_size,
        3 * HEAD_DIM,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    return q, k, v, q_weight, k_weight, cache, positions, output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["case_name"],
        x_vals=CASE_NAMES,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="minimax-h3-qknorm-rope-ulysses-pack",
        args={},
    )
)
def benchmark(case_name: str, provider: str) -> tuple[float, float, float]:
    case = CASE_BY_NAME[case_name]
    q, k, v, q_weight, k_weight, cache, positions, output = make_inputs(case)
    op = MiniMaxH3QKNormRopePackOp()
    implementation = {
        "current": op.forward_cuda,
        "triton": op.forward_triton,
        "cute_dsl": op.forward_cute_dsl,
    }[provider]

    return run_benchmark_no_cudagraph(
        lambda: implementation(
            q,
            k,
            v,
            q_weight,
            k_weight,
            cache,
            positions,
            case.world_size,
            out=output,
        )
    )


if __name__ == "__main__":
    print("Running MiniMax-H3 fused QK-norm/RoPE/Ulysses-pack benchmark...")
    benchmark.run(print_data=True)
