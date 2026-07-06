"""KernelBench L1-40 LayerNorm — 게이트 A seed (신호 다양성 확인용).

문제: Y = layernorm(X, dim=-1) affine 없음, X shape (rows, cols) fp32.
reduction+memory-bound 예상 → 룰 4/6 발화 기대 (softmax와 다른 신호 프로필 노림).
naive Triton (게이트 A는 신호 다양성 확인 목적).

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse

import torch
import triton
import triton.language as tl

GATE_ATOL = 1e-4
GATE_RTOL = 1e-4
GATE_SIZES = (1, 8, 512, 8192)
PROFILE_SIZE = 16384
COLS = 1024
EPS = 1e-5


@triton.jit
def _layernorm_kernel(x_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / n_cols
    y = xc / tl.sqrt(var + eps)
    tl.store(y_ptr + row * n_cols + offs, y, mask=mask)


def solve(X: torch.Tensor, Y: torch.Tensor, rows: int, cols: int):
    BLOCK = triton.next_power_of_2(cols)
    _layernorm_kernel[(rows,)](X, Y, cols, EPS, BLOCK=BLOCK)


def make_case(size, device):
    rows = size
    X = torch.randn(rows, COLS, device=device)
    return {"X": X, "rows": rows, "cols": COLS}


def run_solve(case, device):
    Y = torch.empty(case["rows"], case["cols"], device=device)
    solve(case["X"], Y, case["rows"], case["cols"])
    return Y


def reference(case, device):
    return torch.nn.functional.layer_norm(case["X"], (case["cols"],), eps=EPS)


def _check(device):
    for N in GATE_SIZES:
        case = make_case(N, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        err = (out - ref).abs().max().item()
        print(f"rows={N:>6} {'PASS' if ok else 'FAIL'} max_err={err:.2e}")


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
