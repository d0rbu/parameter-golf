#!/usr/bin/env python3
"""Microbenchmarks and correctness verification for SeedLinear, BottleneckLinear, and CastedLinear.

Usage:
    uv run python benchmark_seed.py                              # full benchmark (needs GPU)
    uv run python benchmark_seed.py --verify                     # correctness checks only
    uv run python benchmark_seed.py --dims 512 --ks 16 64 --compile
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from train_gpt import BottleneckLinear, CastedLinear, SeedLinear

# ---------------------------------------------------------------------------
# Hardware constants (RTX 4090)
# ---------------------------------------------------------------------------
RTX4090_MEM_BW_GBS = 1008.0    # GB/s memory bandwidth
RTX4090_BF16_TFLOPS = 165.0    # TFLOPS bf16


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

def benchmark_layer(
    layer: torch.nn.Module,
    input_shape: tuple[int, ...],
    num_iters: int = 100,
    warmup_iters: int = 10,
    use_compile: bool = False,
) -> dict[str, float]:
    """Benchmark forward+backward time per iteration for a layer.

    Returns dict with median_ms, mean_ms, std_ms, min_ms, max_ms.
    """
    device = next(layer.parameters()).device
    x = torch.randn(input_shape, dtype=torch.bfloat16, device=device, requires_grad=False)

    forward_fn = torch.compile(layer) if use_compile else layer

    # Warmup
    for _ in range(warmup_iters):
        layer.zero_grad(set_to_none=True)
        out = forward_fn(x)
        out.sum().backward()

    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(num_iters):
        layer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = forward_fn(x)
        out.sum().backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)

    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def benchmark_einsum_only(
    num_bases: int,
    out_features: int,
    in_features: int,
    num_iters: int = 100,
    warmup_iters: int = 10,
    device: str = "cuda",
) -> dict[str, float]:
    """Benchmark just the fp8->bf16 cast + einsum operation (no F.linear).

    Returns dict with median_ms and memory bandwidth metrics.
    """
    basis_fp8 = torch.randn(num_bases, out_features, in_features, device=device).to(torch.float8_e4m3fn)
    coeffs = torch.randn(num_bases, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup_iters):
        W = torch.einsum("k,koi->oi", coeffs, basis_fp8.to(torch.bfloat16))
        del W

    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        W = torch.einsum("k,koi->oi", coeffs, basis_fp8.to(torch.bfloat16))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)
        del W

    bw = calc_seed_bandwidth(num_bases, out_features, in_features, statistics.median(times_ms))

    return {
        "median_ms": statistics.median(times_ms),
        "bandwidth_gb_s": bw["bandwidth_gb_s"],
        "utilization_pct": bw["utilization_pct"],
        "total_bytes": bw["total_bytes"],
    }


# ---------------------------------------------------------------------------
# Analytical calculations
# ---------------------------------------------------------------------------

def calc_seed_bandwidth(
    k: int,
    out_features: int,
    in_features: int,
    elapsed_ms: float,
) -> dict[str, float]:
    """Calculate memory bandwidth for the SeedLinear einsum operation.

    Memory traffic breakdown:
      Read  fp8 basis:   k * out * in * 1 byte
      Write bf16 cast:   k * out * in * 2 bytes
      Read  bf16 einsum: k * out * in * 2 bytes
      Write bf16 W:      out * in * 2 bytes
      Total:             k * out * in * 5 + out * in * 2  bytes
    """
    total_bytes = k * out_features * in_features * 5 + out_features * in_features * 2
    elapsed_s = elapsed_ms / 1e3
    bandwidth_gb_s = (total_bytes / 1e9) / elapsed_s if elapsed_s > 0 else 0.0
    utilization_pct = (bandwidth_gb_s / RTX4090_MEM_BW_GBS) * 100.0

    return {
        "total_bytes": total_bytes,
        "bandwidth_gb_s": bandwidth_gb_s,
        "utilization_pct": utilization_pct,
    }


def calc_flops(
    layer_type: str,
    batch: int,
    seq: int,
    out_features: int,
    in_features: int,
    k: int = 0,
    rank: int = 0,
) -> int:
    """Calculate total FLOPs for a single forward pass.

    SeedLinear:     2*k*out*in (einsum) + 2*B*S*out*in (linear)
    BottleneckLinear: 2*B*S*rank*in + 2*B*S*out*rank
    CastedLinear:   2*B*S*out*in
    """
    bs = batch * seq
    if layer_type == "seed":
        return 2 * k * out_features * in_features + 2 * bs * out_features * in_features
    elif layer_type == "bottleneck":
        return 2 * bs * rank * in_features + 2 * bs * out_features * rank
    elif layer_type == "casted":
        return 2 * bs * out_features * in_features
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


# ---------------------------------------------------------------------------
# Correctness verification
# ---------------------------------------------------------------------------

def verify_seed_linear_reproducibility(dims: list[int], ks: list[int], device: str = "cuda") -> None:
    """Create two SeedLinear with same seeds and verify bitwise-identical output."""
    print("--- Verifying SeedLinear reproducibility ---")
    for d in dims:
        for k in ks:
            seeds = torch.arange(k, dtype=torch.int64)
            layer_a = SeedLinear(d, d, k, seeds=seeds).to(device)
            layer_b = SeedLinear(d, d, k, seeds=seeds).to(device)

            # Copy coefficients so outputs should be identical
            layer_b.coeffs.data.copy_(layer_a.coeffs.data)

            x = torch.randn(2, 32, d, dtype=torch.bfloat16, device=device)
            out_a = layer_a(x)
            out_b = layer_b(x)

            match = torch.equal(out_a, out_b)
            status = "PASS" if match else "FAIL"
            print(f"  d={d:4d}  k={k:3d}  [{status}]")
            if not match:
                max_diff = (out_a - out_b).abs().max().item()
                print(f"    max abs diff: {max_diff:.2e}")


def verify_einsum_reference(layer: SeedLinear, x: Tensor) -> bool:
    """Verify layer.forward(x) matches manual einsum + F.linear computation."""
    with torch.no_grad():
        # Layer's own forward
        out_layer = layer(x)

        # Manual reference computation
        coeffs_cast = layer.coeffs.to(x.dtype)
        basis_cast = layer.basis.to(x.dtype)
        W = torch.einsum("k,koi->oi", coeffs_cast, basis_cast)
        out_ref = F.linear(x, W)

    return torch.equal(out_layer, out_ref)


def run_verify(dims: list[int], ks: list[int], device: str = "cuda") -> None:
    """Run all correctness verification tests."""
    print("=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)

    verify_seed_linear_reproducibility(dims, ks, device)

    print()
    print("--- Verifying einsum reference match ---")
    all_pass = True
    for d in dims:
        for k in ks:
            layer = SeedLinear(d, d, k).to(device)
            x = torch.randn(2, 32, d, dtype=torch.bfloat16, device=device)
            match = verify_einsum_reference(layer, x)
            status = "PASS" if match else "FAIL"
            print(f"  d={d:4d}  k={k:3d}  [{status}]")
            if not match:
                all_pass = False
                # Print max diff for diagnosis
                with torch.no_grad():
                    out_actual = layer(x)
                    W = torch.einsum("k,koi->oi", layer.coeffs.to(x.dtype), layer.basis.to(x.dtype))
                    out_ref = F.linear(x, W)
                    print(f"    max abs diff: {(out_actual - out_ref).abs().max().item():.2e}")

    print()
    print("--- Verifying cached forward correctness ---")
    for d in dims:
        for k in ks:
            layer = SeedLinear(d, d, k).to(device)
            x = torch.randn(2, 32, d, dtype=torch.bfloat16, device=device, requires_grad=True)

            # Uncached forward + backward (reference)
            layer._cached_W = None
            out_ref = layer(x)
            out_ref.sum().backward()
            grad_coeffs_ref = layer.coeffs.grad.clone()
            grad_x_ref = x.grad.clone()
            layer.coeffs.grad = None
            x.grad = None

            # Cached forward + backward
            layer.cache_weight()
            out_cached = layer(x)
            out_cached.sum().backward()
            grad_coeffs_cached = layer.coeffs.grad.clone()
            grad_x_cached = x.grad.clone()

            fwd_diff = (out_ref - out_cached).abs().max().item()
            gx_diff = (grad_x_ref - grad_x_cached).abs().max().item()
            gc_diff = (grad_coeffs_ref - grad_coeffs_cached).abs().max().item()
            gc_max = grad_coeffs_ref.abs().max().item()
            gc_rel = gc_diff / gc_max if gc_max > 0 else gc_diff

            ok = fwd_diff < 1e-5 and gx_diff < 1e-5 and gc_rel < 1e-2
            status = "PASS" if ok else "FAIL"
            print(f"  d={d:4d}  k={k:3d}  [{status}]  fwd={fwd_diff:.2e} gx={gx_diff:.2e} gc_rel={gc_rel:.2e}")
            if not ok:
                all_pass = False

    print()
    if all_pass:
        print("All correctness checks passed.")
    else:
        print("SOME CHECKS FAILED -- see above.")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _print_gpu_info() -> None:
    """Print GPU name and peak specs."""
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  ({mem_gb:.1f} GB)")
    print(f"Reference peaks (RTX 4090): {RTX4090_MEM_BW_GBS} GB/s mem BW, {RTX4090_BF16_TFLOPS} TFLOPS bf16")
    print()


def _tflops(fwd_flops: int, elapsed_ms: float) -> float:
    """Forward TFLOPS (fwd FLOPs / total elapsed including backward)."""
    elapsed_s = elapsed_ms / 1e3
    return (fwd_flops / 1e12) / elapsed_s if elapsed_s > 0 else 0.0


def run_benchmarks(
    dims: list[int],
    ks: list[int],
    ranks: list[int],
    batch: int,
    seq: int,
    use_compile: bool,
) -> None:
    """Run all benchmarks and print results tables."""
    device = "cuda"
    input_shape = (batch, seq)

    _print_gpu_info()
    compile_tag = " (torch.compile)" if use_compile else ""

    # ----- SeedLinear -----
    print(f"{'=' * 80}")
    print(f"SeedLinear benchmarks{compile_tag}")
    print(f"  batch={batch}  seq={seq}")
    print(f"{'=' * 80}")
    header = f"{'dim':>6} {'k':>4} {'median':>9} {'mean':>9} {'std':>9} {'TFLOPS':>8} {'BW%':>6}"
    print(header)
    print("-" * len(header))

    for d in dims:
        for k in ks:
            layer = SeedLinear(d, d, k).to(device)
            result = benchmark_layer(layer, (*input_shape, d), use_compile=use_compile)
            flops = calc_flops("seed", batch, seq, d, d, k=k)
            tflops = _tflops(flops, result["median_ms"])
            bw = calc_seed_bandwidth(k, d, d, result["median_ms"])
            print(
                f"{d:>6} {k:>4} {result['median_ms']:>8.3f}ms {result['mean_ms']:>8.3f}ms "
                f"{result['std_ms']:>8.3f}ms {tflops:>7.2f}T {bw['utilization_pct']:>5.1f}%"
            )
            del layer
            torch.cuda.empty_cache()

    # ----- Einsum-only -----
    print()
    print(f"{'=' * 80}")
    print("Einsum-only benchmarks (fp8->bf16 cast + einsum, no F.linear)")
    print(f"{'=' * 80}")
    header = f"{'dim':>6} {'k':>4} {'median':>9} {'BW GB/s':>10} {'BW%':>6} {'MB read':>8}"
    print(header)
    print("-" * len(header))

    for d in dims:
        for k in ks:
            result = benchmark_einsum_only(k, d, d, device=device)
            mb_read = result["total_bytes"] / 1e6
            print(
                f"{d:>6} {k:>4} {result['median_ms']:>8.3f}ms {result['bandwidth_gb_s']:>9.1f} "
                f"{result['utilization_pct']:>5.1f}% {mb_read:>7.1f}MB"
            )

    # ----- BottleneckLinear -----
    print()
    print(f"{'=' * 80}")
    print(f"BottleneckLinear benchmarks{compile_tag}")
    print(f"  batch={batch}  seq={seq}")
    print(f"{'=' * 80}")
    header = f"{'dim':>6} {'rank':>5} {'median':>9} {'mean':>9} {'std':>9} {'TFLOPS':>8}"
    print(header)
    print("-" * len(header))

    for d in dims:
        for r in ranks:
            if r >= d:
                continue  # skip when rank >= dim (not a bottleneck)
            layer = BottleneckLinear(d, d, r).to(device)
            result = benchmark_layer(layer, (*input_shape, d), use_compile=use_compile)
            flops = calc_flops("bottleneck", batch, seq, d, d, rank=r)
            tflops = _tflops(flops, result["median_ms"])
            print(
                f"{d:>6} {r:>5} {result['median_ms']:>8.3f}ms {result['mean_ms']:>8.3f}ms "
                f"{result['std_ms']:>8.3f}ms {tflops:>7.2f}T"
            )
            del layer
            torch.cuda.empty_cache()

    # ----- CastedLinear -----
    print()
    print(f"{'=' * 80}")
    print(f"CastedLinear benchmarks{compile_tag}")
    print(f"  batch={batch}  seq={seq}")
    print(f"{'=' * 80}")
    header = f"{'dim':>6} {'median':>9} {'mean':>9} {'std':>9} {'TFLOPS':>8}"
    print(header)
    print("-" * len(header))

    for d in dims:
        layer = CastedLinear(d, d, bias=False).to(device)
        result = benchmark_layer(layer, (*input_shape, d), use_compile=use_compile)
        flops = calc_flops("casted", batch, seq, d, d)
        tflops = _tflops(flops, result["median_ms"])
        print(
            f"{d:>6} {result['median_ms']:>8.3f}ms {result['mean_ms']:>8.3f}ms "
            f"{result['std_ms']:>8.3f}ms {tflops:>7.2f}T"
        )
        del layer
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmarks and correctness checks for SeedLinear, BottleneckLinear, CastedLinear"
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 768, 1024],
                        help="Hidden dimensions to benchmark (default: 512 768 1024)")
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="Number of seed bases to benchmark (default: 16 32 64 128)")
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256],
                        help="Bottleneck ranks to benchmark (default: 64 128 256)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--seq", type=int, default=1024,
                        help="Sequence length (default: 1024)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for layer benchmarks")
    parser.add_argument("--verify", action="store_true",
                        help="Run correctness verification checks and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmarks require a GPU.")
        return

    if args.verify:
        run_verify(args.dims, args.ks)
        return

    run_benchmarks(
        dims=args.dims,
        ks=args.ks,
        ranks=args.ranks,
        batch=args.batch,
        seq=args.seq,
        use_compile=args.compile,
    )


if __name__ == "__main__":
    main()
