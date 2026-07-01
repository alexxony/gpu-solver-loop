"""2D convolution (single-channel cross-correlation) — compute-bound gain 실험용.

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
문제 출처 = LeetGPU "2D Convolution" (challenge.py). valid padding, stride 1,
cross-correlation (F.conv2d 기본). flat input/kernel → flat output.

왜 이 문제 (다문제 gain 2번째, matmul과 다른 경로):
  out[i,j] = Σ_{p,q} in[i+p, j+q] * ker[p,q].  MAC = out_h*out_w*K^2.
  3072² 입력·15×15 커널 → out 3058², MAC ≈ 2.1G = compute 있음.
  gain 경로 = matmul TF32 아니라 **naive(픽셀당 K² 루프) → 타일링/재사용**.
  = "gain이 TF32 트릭 하나 아님, 다른 커널서도 loop이 개선" 입증용.

seed(이 파일) = naive Triton (픽셀 1개 = 프로그램 1개, K² 내부 루프, 재사용 0).
느린 base. variant가 타일링/blocked로 개선 노림.
"""
import argparse
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False

GATE_ATOL = 1e-3          # fp32 conv 누적오차 (K² 합, 커널값 작음)
GATE_RTOL = 1e-3
# (in_r, in_c, k_r, k_c) 게이트 셋 — 작게. PROFILE = perf test shape.
GATE_SIZES = ((32, 32, 3, 3), (64, 64, 5, 5), (128, 96, 7, 3))
PROFILE_SIZE = (3072, 3072, 15, 15)   # challenge generate_performance_test


def make_case(size, device):
    in_r, in_c, k_r, k_c = size
    g = torch.Generator(device=device).manual_seed(0)
    inp = (torch.rand(in_r * in_c, device=device, dtype=torch.float32, generator=g) * 2.0) - 1.0
    ker = (torch.rand(k_r * k_c, device=device, dtype=torch.float32, generator=g) - 0.5)
    out_r, out_c = in_r - k_r + 1, in_c - k_c + 1
    out = torch.empty(out_r * out_c, device=device, dtype=torch.float32)
    return {"input": inp, "kernel": ker, "output": out,
            "in_r": in_r, "in_c": in_c, "k_r": k_r, "k_c": k_c,
            "out_r": out_r, "out_c": out_c}


if HAS_TRITON:
    @triton.jit
    def _conv_naive_kernel(inp_ptr, ker_ptr, out_ptr,
                           in_c, k_r, k_c, out_r, out_c,
                           n_out, BLOCK: tl.constexpr):
        # 프로그램당 BLOCK개 출력픽셀. 픽셀마다 K² 곱합 (재사용 없음 = 느림).
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out
        oi = offs // out_c          # 출력 행
        oj = offs % out_c           # 출력 열
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for p in range(k_r):
            for q in range(k_c):
                in_idx = (oi + p) * in_c + (oj + q)
                a = tl.load(inp_ptr + in_idx, mask=mask, other=0.0)
                w = tl.load(ker_ptr + p * k_c + q)
                acc += a * w
        tl.store(out_ptr + offs, acc, mask=mask)


def run_solve(case, device):
    inp, ker, out = case["input"], case["kernel"], case["output"]
    n_out = case["out_r"] * case["out_c"]
    BLOCK = 256
    grid = (triton.cdiv(n_out, BLOCK),)
    _conv_naive_kernel[grid](inp, ker, out,
                             case["in_c"], case["k_r"], case["k_c"],
                             case["out_r"], case["out_c"], n_out, BLOCK=BLOCK)
    return out


def reference(case, device):
    # F.conv2d = cross-correlation, valid, stride 1 (challenge reference_impl 동일).
    inp2d = case["input"].view(case["in_r"], case["in_c"]).unsqueeze(0).unsqueeze(0)
    ker2d = case["kernel"].view(case["k_r"], case["k_c"]).unsqueeze(0).unsqueeze(0)
    res = torch.nn.functional.conv2d(inp2d, ker2d, padding=0)
    return res.view(-1)


def _check(device):
    for size in GATE_SIZES:
        case = make_case(size, device)
        out = run_solve(case, device).clone()
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        print(f"{size}: {'PASS' if ok else 'FAIL'} max_err={(out-ref).abs().max():.2e}")


def _bench(device, size=PROFILE_SIZE, iters=30):
    case = make_case(size, device)
    for _ in range(5):
        run_solve(case, device)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        run_solve(case, device)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    mac = case["out_r"] * case["out_c"] * case["k_r"] * case["k_c"]
    print(f"{size}: {ms:.3f}ms  {mac/(ms*1e-3)/1e9:.1f} GMAC/s")


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
