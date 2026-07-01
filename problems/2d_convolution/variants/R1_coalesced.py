"""conv2d variant R1 — 합착(coalesced) 접근. seed의 uncoalesced 개선.

seed(naive): offs = pid*BLOCK + arange → oi=offs//out_c, oj=offs%out_c.
  BLOCK이 출력 행 경계를 넘으면 한 프로그램의 스레드가 입력서 흩어진 행 접근
  = 비합착(load_eff 0.44). 또 인접 출력픽셀이 겹치는 입력을 각자 재로드 = 중복.

R1(합착): 출력을 (행 pid_r, 열블록 pid_c)로 2D 타일링. 한 프로그램 = 한 출력행의
  연속 BLOCK_W개 열. 입력 접근 = 같은 행 in[oi+p, oj+q..oj+q+BLOCK_W] = 연속 =
  워프 합착. 커널 계수는 상수 브로드캐스트. → load_eff↑, 중복로드 L2 히트↑.

= "틀린 가설(fp32 TF32) retire 후 맞는 가설(uncoalesced) 개선" = gain 노림.
계약은 seed와 동일(make_case/run_solve/reference/GATE_*). 커널만 교체.
"""
import argparse
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False

GATE_ATOL = 1e-3
GATE_RTOL = 1e-3
GATE_SIZES = ((32, 32, 3, 3), (64, 64, 5, 5), (128, 96, 7, 3))
PROFILE_SIZE = (3072, 3072, 15, 15)


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
    def _conv_coalesced_kernel(inp_ptr, ker_ptr, out_ptr,
                               in_c, k_r, k_c, out_r, out_c,
                               BLOCK_W: tl.constexpr):
        # 프로그램 = (출력행 oi, 열블록). 한 행의 연속 BLOCK_W 열 = 합착.
        oi = tl.program_id(0)
        col0 = tl.program_id(1) * BLOCK_W
        cols = col0 + tl.arange(0, BLOCK_W)
        mask = (oi < out_r) & (cols < out_c)
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for p in range(k_r):
            row_base = (oi + p) * in_c
            for q in range(k_c):
                # in[oi+p, cols+q] = 연속 주소 → 합착 로드
                a = tl.load(inp_ptr + row_base + (cols + q), mask=mask, other=0.0)
                w = tl.load(ker_ptr + p * k_c + q)
                acc += a * w
        tl.store(out_ptr + oi * out_c + cols, acc, mask=mask)


def run_solve(case, device):
    inp, ker, out = case["input"], case["kernel"], case["output"]
    BLOCK_W = 128
    grid = (case["out_r"], triton.cdiv(case["out_c"], BLOCK_W))
    _conv_coalesced_kernel[grid](inp, ker, out,
                                 case["in_c"], case["k_r"], case["k_c"],
                                 case["out_r"], case["out_c"], BLOCK_W=BLOCK_W)
    return out


def reference(case, device):
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
