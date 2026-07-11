"""kb_matmul_scalar R_tf32 — `fp32_no_tensorcore` 처방 충실판 (null 예상).

처방 = TF32 허용(torch.backends). 단 이 커널은 hand-Triton **scalar 루프**라
cuBLAS/tl.dot 경로가 없음 → TF32 스위치는 실측 latency에 무효 = **misfire 입증용 null**.
ON 트랙: 이 variant가 개선 없음(improved=False) 반복 → fp32 룰 demote→retire.

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
    # seed와 동일: program 하나가 C의 BLOCK_M×BLOCK_N 타일, K는 scalar 루프.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K):
        a = tl.load(a_ptr + offs_m[:, None] * K + k,
                    mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptr + k * N + offs_n[None, :],
                    mask=offs_n[None, :] < N, other=0.0)
        acc += a * b
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
    # fp32_no_tensorcore 처방(TF32 허용)을 커널 실행 스코프에만 적용 후 복원 —
    # 전역으로 켜면 게이트 reference(A@B, cuBLAS)까지 TF32 오염돼 correctness 실패.
    # scalar Triton 커널은 cuBLAS/tl.dot 경로가 없어 이 처방은 latency 무효 = null.
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        C = torch.empty(case["M"], case["N"], device=device)
        solve(case["A"], case["B"], C, case["M"], case["N"], case["K"])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
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
