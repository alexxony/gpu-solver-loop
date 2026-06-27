"""Group Normalization — LeetGPU challenge. Triton reduction 커널.

문제: Y = group_norm(X, gamma, beta, G, eps). X shape (N,C,H,W) fp32.
그룹 단위(C/G 채널 × H×W 원소)로 mean/var → 정규화 → per-channel affine. atol/rtol 1e-3.
self-contained — executor gate가 reference()와 교차검증 (challenge.py import 안 함).

reduction = sigmoid(순수 elementwise)와 다른 신호 프로필 기대 (그룹별 합산 → L2/occupancy 특성).

사용: python solve.py --check | --bench | --profile
executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
from __future__ import annotations
import argparse

import torch
import triton
import triton.language as tl

GATE_ATOL = 1e-3
GATE_RTOL = 1e-3
# (N, C, H, W, G) — 작은/중간/큰 케이스. reduction이라 그룹크기 변화 커버.
GATE_SIZES = (
    (1, 4, 1, 1, 2),         # 최소 — 그룹당 2채널, 공간 1
    (2, 8, 4, 4, 4),         # 작은 4D
    (4, 32, 16, 16, 8),      # 중간
    (8, 64, 32, 32, 16),     # 큰
)
# 챔피언 측정용 = 큰 4D 텐서 (충분한 reduction 작업량).
PROFILE_SIZE = (16, 128, 64, 64, 32)
EPS = 1e-5


@triton.jit
def _group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    C, HW, cpg, group_elems,           # cpg=C/G 채널, group_elems=cpg*HW
    eps,
    BLOCK: tl.constexpr,
):
    # 프로그램 = (n, g) 하나. 그룹 한 개의 group_elems 원소 정규화.
    pid = tl.program_id(0)              # = n * G + g
    n = pid // (C // cpg)
    g = pid % (C // cpg)
    # 그룹 첫 원소의 flat offset (X는 (N,C,H,W) row-major).
    group_start = n * C * HW + g * cpg * HW

    # pass 1 — 그룹 mean/var (group_elems 원소 합산).
    mean = 0.0
    m2 = 0.0
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean = mean / group_elems
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        d = tl.where(mask, x - mean, 0.0)
        m2 += tl.sum(d * d, axis=0)
    var = m2 / group_elems
    rstd = 1.0 / tl.sqrt(var + eps)

    # pass 2 — 정규화 + per-channel affine. 채널 = (g*cpg + local_c).
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        local_c = idx // HW                # 그룹 내 채널 인덱스 0..cpg-1
        ch = g * cpg + local_c             # 전역 채널 인덱스
        gw = tl.load(gamma_ptr + ch, mask=mask, other=0.0)
        bb = tl.load(beta_ptr + ch, mask=mask, other=0.0)
        y = (x - mean) * rstd * gw + bb
        tl.store(y_ptr + group_start + idx, y, mask=mask)


def solve(X, gamma, beta, Y, N, C, H, W, G, eps):
    # [variant: fp32_no_tensorcore 가설 충실] TF32 허용 — fp32 matmul을 텐서코어로.
    # ⚠️ 의도된 null: groupnorm은 Triton reduction 커널, matmul 없음 → 효과 0.
    #   발화 룰(fp32_no_tensorcore)이 이 워크로드에 오발화임을 latency 불변으로 폭로.
    torch.set_float32_matmul_precision("high")
    HW = H * W
    cpg = C // G
    group_elems = cpg * HW
    grid = (N * G,)
    # BLOCK = group_elems 덮을 2의 거듭제곱 (단일 블록 reduction).
    BLOCK = max(triton.next_power_of_2(group_elems), 16)
    _group_norm_kernel[grid](
        X, gamma, beta, Y,
        C, HW, cpg, group_elems,
        eps,
        BLOCK=BLOCK,
    )


# ── executor 통일 어댑터 계약 ──
def _make_case(shape, device, seed=0):
    N, C, H, W, G = shape
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.empty(N, C, H, W, device=device).uniform_(-3.0, 3.0, generator=g)
    gamma = torch.empty(C, device=device).uniform_(0.5, 1.5, generator=g)
    beta = torch.empty(C, device=device).uniform_(-0.5, 0.5, generator=g)
    return {"X": X, "gamma": gamma, "beta": beta,
            "N": N, "C": C, "H": H, "W": W, "G": G}


def make_case(size, device):
    return _make_case(size, device)


def run_solve(case, device):
    Y = torch.empty_like(case["X"])
    solve(case["X"], case["gamma"], case["beta"], Y,
          case["N"], case["C"], case["H"], case["W"], case["G"], EPS)
    return Y


def reference(case, device):
    return torch.nn.functional.group_norm(
        case["X"], case["G"], weight=case["gamma"], bias=case["beta"], eps=EPS)


def _check(device):
    for shape in GATE_SIZES:
        case = make_case(shape, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        err = (out - ref).abs().max().item()
        print(f"shape={shape} {'PASS' if ok else 'FAIL'} max_err={err:.2e}")


def _bench(device, shape=PROFILE_SIZE, iters=50):
    case = make_case(shape, device)
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
    print(f"shape={shape} latency={s.elapsed_time(e)/iters*1000:.1f}us")


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
