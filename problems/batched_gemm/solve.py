"""batched GEMM — B개 독립 fp32 정사각행렬곱 동시 수행. compute-bound gain 다문제화 후보.

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
self-contained — reference()와 교차검증.

왜 이 문제: matmul(단일 4096²)의 gain 재현형. FLOP=O(B*N^3) 유지하며 배치 차원만 추가
→ compute-bound 성질 그대로(matmul과 동일 이유) + KernelBench/실무에서 흔한 batched GEMM
패턴 커버. conv2d(gain null, ALU 천장)와 달리 matmul처럼 TF32 텐서코어 경로 열려있어
gain 재현 유력 후보 (PROGRESS.md 2026-07-11 "gain 다문제화" 세션 선택지 3).

seed(이 파일) = TF32 OFF (느린 base). variant = TF32 ON (variants/R1.py).
"""
import argparse
import torch

GATE_ATOL = 6e-2                  # TF32 누적오차 수용 — matmul seed와 동일 근거(가수 10비트)
GATE_RTOL = 6e-2
GATE_SIZES = (128, 512, 1024)      # (B, N) 아님 — 배치 고정 GATE_B, 여기는 N만 스윕
GATE_B = 8
PROFILE_B = 16
PROFILE_SIZE = 1024                # (16, 1024, 1024) batched → 16*2*1024^3 = 34.4 GFLOP compute-bound

# seed = TF32 OFF (느린 fp32 코어). variant가 이걸 ON으로 켜서 gain 노림.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _make_case(B, N, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(B, N, N, device=device, dtype=torch.float32, generator=g)
    b = torch.randn(B, N, N, device=device, dtype=torch.float32, generator=g)
    return a, b


def make_case(size, device):
    # size = N (게이트 스윕용). 게이트는 GATE_B, 프로파일은 PROFILE_B 사용.
    B = PROFILE_B if size == PROFILE_SIZE else GATE_B
    a, b = _make_case(B, size, device)
    return {"A": a, "B_": b, "N": size, "B": B}


def run_solve(case, device):
    return torch.bmm(case["A"], case["B_"])


def reference(case, device):
    # 레퍼런스 = fp64 누적 (정확) → fp32 결과와 완화 atol로 비교.
    return torch.bmm(case["A"].double(), case["B_"].double()).float()


def _check(device):
    for N in GATE_SIZES:
        case = make_case(N, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        print(f"N={N} B={case['B']}: {'PASS' if ok else 'FAIL'} max_err={(out-ref).abs().max():.2e}")


def _bench(device, N=PROFILE_SIZE, iters=50):
    case = make_case(N, device)
    for _ in range(10):
        run_solve(case, device)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        run_solve(case, device)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    flop = case["B"] * 2 * N**3
    print(f"B={case['B']} N={N}: {ms:.3f}ms  {flop/(ms*1e-3)/1e12:.1f} TFLOP/s  tf32={torch.backends.cuda.matmul.allow_tf32}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.check:
        _check(dev)
    elif a.bench:
        _bench(dev)
    elif a.profile:
        run_solve(make_case(PROFILE_SIZE, dev), dev)   # ncu/Event용 1 forward
