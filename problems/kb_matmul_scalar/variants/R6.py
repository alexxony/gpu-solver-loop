"""KernelBench L1-1 계열 Matmul (scalar naive) — 진화 다문제 seed (conv2d retire 재현형).

문제: C = A @ B, A(M,K) B(K,N) fp32. **scalar 명시 루프**(tl.dot 안 씀).
목적: tl.dot은 TF32 텐서코어 자동(tc=True)이라 fp32_no_tensorcore 룰 안 뜸.
scalar 루프 = tc=False 유지 → `fp32_no_tensorcore` 발화하나 **이 커널엔 TF32 무효**
(cuBLAS sgemm 아니라 hand-Triton scalar) = misfire → flat 큐로 retire 강제.
= conv2d의 fp32 misfire→retire를 matmul로 재현 = 진화 축 3문제.

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse

import torch
import triton
import triton.language as tl

GATE_ATOL = 1e-2
GATE_RTOL = 1e-2
GATE_SIZES = (16, 64, 256, 512)   # size = M=N=K (정사각)
PROFILE_SIZE = 512


@triton.jit
def _matmul_scalar_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # program 하나가 C의 BLOCK_M×BLOCK_N 타일. K는 scalar 루프(tl.dot 회피).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K):
        a = tl.load(a_ptr + offs_m[:, None] * K + k,
                    mask=offs_m[:, None] < M, other=0.0)          # (BLOCK_M,1)
        b = tl.load(b_ptr + k * N + offs_n[None, :],
                    mask=offs_n[None, :] < N, other=0.0)          # (1,BLOCK_N)
        acc += a * b                                              # outer product, no tl.dot
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_scalar_kernel[grid](A, B, C, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


def make_case(size, device):
    M = N = K = size
    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)
    return {"A": A, "B": B, "M": M, "N": N, "K": K}


def run_solve(case, device):
    C = torch.empty(case["M"], case["N"], device=device)
    solve(case["A"], case["B"], C, case["M"], case["N"], case["K"])
    return C


def reference(case, device):
    return case["A"] @ case["B"]


def _check(device):
    for N in GATE_SIZES:
        case = make_case(N, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        err = (out - ref).abs().max().item()
        print(f"size={N:>5} {'PASS' if ok else 'FAIL'} max_err={err:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.profile:
        case = make_case(PROFILE_SIZE, dev)
        run_solve(case, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
    else:
        _check(dev)
