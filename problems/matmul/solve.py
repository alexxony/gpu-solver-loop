"""matmul — 큰 정사각 fp32 행렬곱. compute-bound gain 실험용.

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
self-contained — reference()와 교차검증.

왜 이 문제: FLOP=O(N^3), mem=O(N^2) → N 크면 compute-bound. fp32 matmul은
기본 FP32 코어(느림). TF32/bf16 켜면 텐서코어 → 진짜 speedup (PoC R5 1.71× 효과).
= 진화 가설(fp32_no_tensorcore → TF32) 이 실제 latency 가르는 문제.

seed(이 파일) = TF32 OFF (느린 base). variant = TF32 ON (variants/R_tf32on.py).
"""
import argparse
import torch

GATE_ATOL = 6e-2                  # TF32 4096² 누적오차 수용 (실측 max_err≈0.045, 가수 10비트)
GATE_RTOL = 6e-2
# 작은 게이트 + 큰 PROFILE. compute-bound 위해 PROFILE 크게.
GATE_SIZES = (128, 512, 1024)
PROFILE_SIZE = 4096               # 4096^3 = 68.7 GFLOP → A100서 compute-bound

# seed = TF32 OFF (느린 fp32 코어). variant가 이걸 ON으로 켜서 gain 노림.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _make_case(N, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(N, N, device=device, dtype=torch.float32, generator=g)
    b = torch.randn(N, N, device=device, dtype=torch.float32, generator=g)
    return a, b


def make_case(size, device):
    a, b = _make_case(size, device)
    return {"A": a, "B": b, "N": size}


def run_solve(case, device):
    return case["A"] @ case["B"]


def reference(case, device):
    # 레퍼런스 = fp64 누적 (정확) → fp32 결과와 완화 atol로 비교.
    return (case["A"].double() @ case["B"].double()).float()


def _check(device):
    for N in GATE_SIZES:
        case = make_case(N, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        print(f"N={N}: {'PASS' if ok else 'FAIL'} max_err={(out-ref).abs().max():.2e}")


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
    flop = 2 * N**3
    print(f"N={N}: {ms:.3f}ms  {flop/(ms*1e-3)/1e12:.1f} TFLOP/s  tf32={torch.backends.cuda.matmul.allow_tf32}")


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
