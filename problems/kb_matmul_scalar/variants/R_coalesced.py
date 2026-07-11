"""kb_matmul_scalar R_coalesced — `uncoalesced` 처방 충실판 (실측 gain 기대).

seed 문제(probe 실측): A 로드가 k당 (BLOCK_M,1) stride-K 세로 조각 → load_eff 41.67%.
처방 = K를 BK 청크로 타일해 A(BM,BK)·B(BK,BN)를 **합착 블록 로드** 후 블록 MAC.
MAC은 `tl.dot(..., input_precision="ieee")` = FFMA 강제(TF32/텐서코어 배제)
→ gain의 인과가 합착/타일링에만 귀속 (probe: tc_pct 0.02%≈0, 10.1× @A100).

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
def _matmul_coalesced_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # K를 BK 청크로 타일: a(BM,BK)·b(BK,BN) 합착 블록 로드 → ieee dot(FFMA) 누적.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")   # FFMA — TC/TF32 배제
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    BM, BN, BK = 32, 32, 64                      # probe 최속 구성 (BK 스윕 16/32/64)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _matmul_coalesced_kernel[grid](A, B, C, M, N, K, BM=BM, BN=BN, BK=BK)


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
