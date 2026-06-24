"""Sigmoid Activation — LeetGPU challenge. Triton elementwise 커널.

문제: Y = sigmoid(X), X/Y fp32 길이 N. atol/rtol 1e-5.
self-contained — executor gate가 reference()와 교차검증 (challenge.py import 안 함).

사용: python solve.py --check | --bench | --profile
executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse

import torch
import triton
import triton.language as tl

GATE_ATOL = 1e-5
GATE_RTOL = 1e-5
# elementwise — 작은/중간/큰 길이로 게이트. 챔피언 측정은 PROFILE_SIZE.
GATE_SIZES = (1, 4, 1024, 100000)
PROFILE_SIZE = 50_000_000          # challenge generate_performance_test와 동일


@triton.jit
def _sigmoid_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))       # sigmoid
    tl.store(y_ptr + offs, y, mask=mask)


def solve(X: torch.Tensor, Y: torch.Tensor, N: int):
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _sigmoid_kernel[grid](X, Y, N, BLOCK)


# ── executor 통일 어댑터 계약 ──
def _make_case(N, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.empty(N, device=device).uniform_(-10.0, 10.0, generator=g)
    return X


def make_case(size, device):
    return {"X": _make_case(size, device), "N": size}


def run_solve(case, device):
    Y = torch.empty(case["N"], device=device)
    solve(case["X"], Y, case["N"])
    return Y


def reference(case, device):
    return torch.sigmoid(case["X"])


def _check(device):
    for N in GATE_SIZES:
        case = make_case(N, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        err = (out - ref).abs().max().item()
        print(f"N={N:>9} {'PASS' if ok else 'FAIL'} max_err={err:.2e}")


def _bench(device, N=PROFILE_SIZE, iters=50):
    case = make_case(N, device)
    if device == "cpu":
        print("CPU — latency 의미 없음"); return
    for _ in range(10):
        run_solve(case, device)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        run_solve(case, device)
    e.record()
    torch.cuda.synchronize()
    print(f"N={N} latency={s.elapsed_time(e)/iters*1000:.1f}us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--profile", action="store_true")  # ncu가 이 경로 실행
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.check:
        _check(dev)
    elif args.bench:
        _bench(dev)
    elif args.profile:
        # ncu가 측정할 단일 커널 실행 (1회).
        case = make_case(PROFILE_SIZE, dev)
        run_solve(case, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
    else:
        _check(dev)
