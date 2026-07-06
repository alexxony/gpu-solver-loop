"""KernelBench L1-47 Sum Reduction — 게이트 A seed (신호 다양성 확인용).

문제: Y = sum(X, dim=-1), X shape (rows, cols) fp32 → Y shape (rows,).
row당 reduction. bandwidth/occupancy 신호가 softmax/norm과 다를 것 기대
(출력 작음, 입력만 큼 = 순수 read-reduction). 룰 6/8/9 갈림 노림.
naive Triton (게이트 A는 신호 다양성 확인 목적).

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse

import torch
import triton
import triton.language as tl

GATE_ATOL = 1e-3
GATE_RTOL = 1e-3
GATE_SIZES = (1, 8, 512, 8192)
PROFILE_SIZE = 16384
COLS = 4096


@triton.jit
def _reduce_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(y_ptr + row, acc)


def solve(X: torch.Tensor, Y: torch.Tensor, rows: int, cols: int):
    BLOCK = 1024
    _reduce_kernel[(rows,)](X, Y, cols, BLOCK=BLOCK)


def make_case(size, device):
    rows = size
    X = torch.randn(rows, COLS, device=device)
    return {"X": X, "rows": rows, "cols": COLS}


def run_solve(case, device):
    Y = torch.empty(case["rows"], device=device)
    solve(case["X"], Y, case["rows"], case["cols"])
    return Y


def reference(case, device):
    return case["X"].sum(dim=-1)


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
