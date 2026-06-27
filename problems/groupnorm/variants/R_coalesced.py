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


# [variant: uncoalesced/occupancy 가설 충실] baseline 진짜 병목:
#   grid=N*G=512 그룹당 1프로그램 → group_elems(16384) 통째 단일블록 순회.
#   occupancy 0.51 = SM 절반 놀고. BLOCK 줄여도 grid 작아 병렬도 안 늚.
# 변경 = 그룹을 NTILE 블록으로 쪼개 grid=N*G*NTILE 로 병렬도↑.
#   타일별 부분합 atomic 누적(커널1) → finalize(커널2) → 정규화(커널3).
#   타일 내 연속 접근 = 합착 로드 복구. 진짜 다른 알고리즘 (단일블록→분할 reduction).

@triton.jit
def _partial_kernel(x_ptr, sum_ptr, sumsq_ptr,
                    cpg, HW, group_elems, NTILE,
                    BLOCK: tl.constexpr):
    # 프로그램 = (그룹 gid, 타일 tid). 그룹의 group_elems를 NTILE 등분, 타일 1조각 합산.
    gid = tl.program_id(0)          # = n*G + g
    tid = tl.program_id(1)          # 0..NTILE-1
    group_start = gid * group_elems
    tile = (group_elems + NTILE - 1) // NTILE
    base = tid * tile
    tile_end = base + tile        # 타일 상한 (다음 타일 침범 방지)
    s = 0.0
    ss = 0.0
    for off in range(0, tile, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        # mask = 그룹 끝 AND 타일 끝. BLOCK>tile이면 arange가 다음 타일 침범 →
        # idx<tile_end 없으면 작은 그룹서 타일 간 중복 합산 (max_err=1.43 버그).
        mask = (idx < group_elems) & (idx < tile_end)
        x = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)
    tl.atomic_add(sum_ptr + gid, s)
    tl.atomic_add(sumsq_ptr + gid, ss)


@triton.jit
def _finalize_kernel(sum_ptr, sumsq_ptr, mean_ptr, rstd_ptr,
                     group_elems, eps, NG):
    gid = tl.program_id(0)
    mask = gid < NG
    s = tl.load(sum_ptr + gid, mask=mask, other=0.0)
    ss = tl.load(sumsq_ptr + gid, mask=mask, other=0.0)
    mean = s / group_elems
    var = ss / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + gid, mean, mask=mask)
    tl.store(rstd_ptr + gid, rstd, mask=mask)


@triton.jit
def _norm_affine_kernel(x_ptr, gamma_ptr, beta_ptr, y_ptr,
                        mean_ptr, rstd_ptr,
                        C, HW, cpg, group_elems, G, NTILE,
                        BLOCK: tl.constexpr):
    gid = tl.program_id(0)         # = n*G + g
    tid = tl.program_id(1)
    g = gid % G                    # 그룹 인덱스 (채널은 n 무관 반복)
    group_start = gid * group_elems
    mean = tl.load(mean_ptr + gid)
    rstd = tl.load(rstd_ptr + gid)
    tile = (group_elems + NTILE - 1) // NTILE
    base = tid * tile
    tile_end = base + tile        # 타일 상한 (partial과 동일 — 다음 타일 침범 방지)
    for off in range(0, tile, BLOCK):
        idx = base + off + tl.arange(0, BLOCK)
        mask = (idx < group_elems) & (idx < tile_end)
        x = tl.load(x_ptr + group_start + idx, mask=mask, other=0.0)
        local_c = idx // HW
        ch = g * cpg + local_c
        gw = tl.load(gamma_ptr + ch, mask=mask, other=0.0)
        bb = tl.load(beta_ptr + ch, mask=mask, other=0.0)
        tl.store(y_ptr + group_start + idx,
                 (x - mean) * rstd * gw + bb, mask=mask)


def solve(X, gamma, beta, Y, N, C, H, W, G, eps):
    HW = H * W
    cpg = C // G
    group_elems = cpg * HW
    NG = N * G
    NTILE = 8                      # 그룹당 8블록 → grid 512→4096 = 병렬도 8배
    BLOCK = 1024
    sum_ = torch.zeros(NG, device=X.device, dtype=torch.float32)
    sumsq = torch.zeros(NG, device=X.device, dtype=torch.float32)
    mean = torch.empty(NG, device=X.device, dtype=torch.float32)
    rstd = torch.empty(NG, device=X.device, dtype=torch.float32)

    _partial_kernel[(NG, NTILE)](X, sum_, sumsq, cpg, HW, group_elems, NTILE, BLOCK=BLOCK)
    _finalize_kernel[(NG,)](sum_, sumsq, mean, rstd, group_elems, eps, NG)
    # 정규화 + affine — gamma/beta는 per-channel. 그룹 gid → 채널 (gid%G)*cpg + local_c.
    _norm_affine_kernel[(NG, NTILE)](
        X, gamma, beta, Y, mean, rstd,
        C, HW, cpg, group_elems, G, NTILE, BLOCK=BLOCK)


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
