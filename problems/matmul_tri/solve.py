"""matmul_tri — 손수 Triton matmul. 두 축 동시 입증용 (진화 retire + gain).

seed(이 파일) = **비합착** Triton matmul. B를 비합착 접근(stride 큰 축 먼저) →
load_eff↓ → `uncoalesced` 룰 조건. 단 fp32+compute+not tensorcore라 priority 1
`fp32_no_tensorcore`가 먼저 발화 → TF32 variant. **Triton 커널은 set_float32_matmul_precision
무효**(cuBLAS 아님) → null → demote 누적 → retire → `uncoalesced` 승격 → 합착 fix → 진짜 gain.

= ON: fp32룰 헛발화→retire→uncoalesced→합착→빨라짐. OFF: fp32룰 영원→TF32 무효 영원.
  ON best < OFF best = 진화가 gain 낸다 (두 축 한 문제서 동시).

executor 계약: make_case / run_solve / reference / GATE_SIZES / PROFILE_SIZE.
"""
import argparse
import torch
import triton
import triton.language as tl

GATE_ATOL = 1e-1                  # fp32 Triton matmul 누적오차 (큰 N)
GATE_RTOL = 1e-1
GATE_SIZES = (256, 512)
PROFILE_SIZE = 2048               # 손수 커널이라 작게 (검증 속도)

torch.backends.cuda.matmul.allow_tf32 = False


@triton.jit
def _mm_uncoalesced(a_ptr, b_ptr, c_ptr, M, N, K,
                    sm, sk, skb, snb, scm, scn, BM: tl.constexpr,
                    BN: tl.constexpr, BK: tl.constexpr):
    # ponytail: 비합착 — B를 (k,n)서 n-major로 읽어 워프가 stride 큰 축 횡단.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + offs_m[:, None] * sm + (k + offs_k)[None, :] * sk,
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        # 비합착: B 인덱스를 n*skb + k*snb (축 뒤바꿈) → load_eff 저하
        b = tl.load(b_ptr + (k + offs_k)[:, None] * snb + offs_n[None, :] * skb,
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _matmul_tri(a, b, BM=64, BN=64, BK=32):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    # 비합착: B의 stride를 일부러 뒤바꿔 전달 (snb=col stride, skb=row stride)
    _mm_uncoalesced[grid](a, b, c, M, N, K,
                          a.stride(0), a.stride(1), b.stride(1), b.stride(0),
                          c.stride(0), c.stride(1), BM, BN, BK)
    return c


def _make_case(N, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(N, N, device=device, dtype=torch.float32, generator=g)
    b = torch.randn(N, N, device=device, dtype=torch.float32, generator=g)
    return a, b


def make_case(size, device):
    a, b = _make_case(size, device)
    return {"A": a, "B": b, "N": size}


def run_solve(case, device):
    return _matmul_tri(case["A"], case["B"])


def reference(case, device):
    return (case["A"].double() @ case["B"].double()).float()


def _check(device):
    for N in GATE_SIZES:
        case = make_case(N, device)
        out = run_solve(case, device)
        ref = reference(case, device)
        ok = torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
        print(f"N={N}: {'PASS' if ok else 'FAIL'} max_err={(out-ref).abs().max():.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.check:
        _check(dev)
    elif a.profile:
        run_solve(make_case(PROFILE_SIZE, dev), dev)
