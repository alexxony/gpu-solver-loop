"""batched_gemm variant — TF32 ON. seed(OFF) 대비 텐서코어 사용 → speedup 노림.

seed와 유일 차이 = allow_tf32=True. matmul R_tf32on.py와 동일 패턴.
"""
import argparse
import torch

GATE_ATOL = 6e-2
GATE_RTOL = 6e-2
GATE_SIZES = (128, 512, 1024)
GATE_B = 8
PROFILE_B = 16
PROFILE_SIZE = 1024

# variant 핵심: TF32 ON → bmm이 텐서코어 경로
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _make_case(B, N, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(B, N, N, device=device, dtype=torch.float32, generator=g)
    b = torch.randn(B, N, N, device=device, dtype=torch.float32, generator=g)
    return a, b


def make_case(size, device):
    B = PROFILE_B if size == PROFILE_SIZE else GATE_B
    a, b = _make_case(B, size, device)
    return {"A": a, "B_": b, "N": size, "B": B}


def run_solve(case, device):
    return torch.bmm(case["A"], case["B_"])


def reference(case, device):
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
        run_solve(make_case(PROFILE_SIZE, dev), dev)
