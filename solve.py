"""R1 champion — Llama decoder block, flash attention (PoC 01-hard-loop-poc).

R0 naive (score (8,T,T) materialise, 3.495ms) → R1 flash (1.723ms, 2.03×).
변형 = attention 6줄만 F.scaled_dot_product_attention(is_causal)로 교체.
RMSNorm/RoPE/GQA/FFN 동일. PoC 챔피언 (R2/R2' 반증, R1 유지).

단일소스 (.py, ipynb 폐기 — PoC §110). 글루 Generator의 시드 코드로도 쓰임.
사용: python solve.py --check | --bench | --profile

weight 레이아웃 = challenge.py 오프셋과 일치:
  rms1(512) WQ(512,512) WK(128,512) WV(128,512) WO(512,512)
  rms2(512) Wg(1408,512) Wu(1408,512) Wd(512,1408)  → TOTAL 2,819,072
"""
import argparse
import torch
import torch.nn.functional as F

D = 512
NQ, NKV, HD = 8, 2, 64
FF = 1408
EPS = 1e-5

# offsets (floats) — challenge.py O_* 와 일치
O_RMS1 = 0
O_WQ = O_RMS1 + D                 # 512
O_WK = O_WQ + (NQ * HD) * D       # 262656
O_WV = O_WK + (NKV * HD) * D      # 328192
O_WO = O_WV + (NKV * HD) * D      # 393728
O_RMS2 = O_WO + D * D             # 655872
O_WG = O_RMS2 + D                 # 656384
O_WU = O_WG + FF * D              # 1377280
O_WD = O_WU + FF * D              # 2098176
TOTAL = O_WD + D * FF             # 2819072


def _rmsnorm(z, w):
    var = z.pow(2).mean(dim=-1, keepdim=True)
    return z * torch.rsqrt(var + EPS) * w


def _rope(t, cos, sin):
    # t:(T,H,64) cos/sin:(T,32) broadcast over heads
    t1, t2 = t[..., :32], t[..., 32:]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat([t1 * c - t2 * s, t1 * s + t2 * c], dim=-1)


def solve(x, output, weights, cos, sin, seq_len):
    T = seq_len

    def W(off, r, c):
        return weights[off:off + r * c].view(r, c)

    w1 = weights[O_RMS1:O_WQ]
    WQ = W(O_WQ, NQ * HD, D)
    WK = W(O_WK, NKV * HD, D)
    WV = W(O_WV, NKV * HD, D)
    WO = W(O_WO, D, D)
    w2 = weights[O_RMS2:O_WG]
    Wg = W(O_WG, FF, D)
    Wu = W(O_WU, FF, D)
    Wd = W(O_WD, D, FF)

    # ---- Attention (pre-norm) ----
    h = _rmsnorm(x, w1)
    q = (h @ WQ.t()).view(T, NQ, HD)
    k = (h @ WK.t()).view(T, NKV, HD)
    v = (h @ WV.t()).view(T, NKV, HD)
    q = _rope(q, cos, sin)
    k = _rope(k, cos, sin)

    # GQA broadcast → (T,8,64)
    k = k.repeat_interleave(NQ // NKV, dim=1)
    v = v.repeat_interleave(NQ // NKV, dim=1)

    # (8,T,64)
    qh = q.transpose(0, 1)
    kh = k.transpose(0, 1)
    vh = v.transpose(0, 1)

    # ★R1: flash attention — score (8,T,T) materialise 제거 (R0 6줄 대체)
    ctx = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)  # (8,T,64)
    ctx = ctx.transpose(0, 1).reshape(T, D)

    attn_out = ctx @ WO.t()
    x1 = x + attn_out

    # ---- FFN (pre-norm, SwiGLU) ----
    h2 = _rmsnorm(x1, w2)
    gate = h2 @ Wg.t()
    up = h2 @ Wu.t()
    act = F.silu(gate) * up      # ← R3 융합 타깃 (ncu 126μs, 메모리바운드)
    ffn = act @ Wd.t()
    x2 = x1 + ffn
    output.copy_(x2)


# ---- harness: 독립 참조 (challenge.py 없이도 자체 검증) ----

def _reference(x, weights, cos, sin, T):
    """다른 코드 경로 ref (GQA expand+명시, masked_fill softmax). solve와 교차검증."""
    def W(off, r, c):
        return weights[off:off + r * c].view(r, c)
    w1 = weights[O_RMS1:O_WQ]; WQ = W(O_WQ, NQ*HD, D); WK = W(O_WK, NKV*HD, D)
    WV = W(O_WV, NKV*HD, D); WO = W(O_WO, D, D); w2 = weights[O_RMS2:O_WG]
    Wg = W(O_WG, FF, D); Wu = W(O_WU, FF, D); Wd = W(O_WD, D, FF)
    h = _rmsnorm(x, w1)
    q = _rope((h @ WQ.t()).view(T, NQ, HD), cos, sin)
    k = _rope((h @ WK.t()).view(T, NKV, HD), cos, sin)
    v = (h @ WV.t()).view(T, NKV, HD)
    qh = q.transpose(0, 1)                                   # (8,T,64)
    kh = k.transpose(0, 1).unsqueeze(1).expand(NKV, NQ//NKV, T, HD).reshape(NQ, T, HD)
    vh = v.transpose(0, 1).unsqueeze(1).expand(NKV, NQ//NKV, T, HD).reshape(NQ, T, HD)
    sc = (qh @ kh.transpose(-1, -2)) / (HD ** 0.5)
    m = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), 1)
    ctx = (torch.softmax(sc + m, dim=-1) @ vh).transpose(0, 1).reshape(T, D)
    x1 = x + ctx @ WO.t()
    h2 = _rmsnorm(x1, w2)
    return x1 + (F.silu(h2 @ Wg.t()) * (h2 @ Wu.t())) @ Wd.t()


def _make_case(T, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    weights = torch.empty(TOTAL, device=device).normal_(0, 0.02, generator=g)
    weights[O_RMS1:O_WQ].uniform_(0.8, 1.2, generator=g)
    weights[O_RMS2:O_WG].uniform_(0.8, 1.2, generator=g)
    x = torch.empty(T, D, device=device).uniform_(-1, 1, generator=g)
    pos = torch.arange(T, device=device, dtype=torch.float32)
    freq = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, device=device).float() / HD))
    ang = torch.outer(pos, freq)
    return x, weights, ang.cos(), ang.sin()


def _check(device):
    ok = True
    for T in (1, 4, 128, 2048):
        x, w, cos, sin = _make_case(T, device)
        out = torch.empty(T, D, device=device)
        solve(x, out, w, cos, sin, T)
        ref = _reference(x, w, cos, sin, T)
        err = (out - ref).abs().max().item()
        passed = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
        ok &= passed
        print(f"  T={T:5d}  max_err={err:.2e}  {'PASS' if passed else 'FAIL'}")
    print("check:", "PASS" if ok else "FAIL")
    return ok


def _bench(device, T=2048, iters=50):
    x, w, cos, sin = _make_case(T, device)
    out = torch.empty(T, D, device=device)
    for _ in range(10):
        solve(x, out, w, cos, sin, T)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        solve(x, out, w, cos, sin, T)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    print(f"bench T={T}: {ms:.3f} ms/iter  (R0 naive 3.495 / R1 flash 1.723 기준)")
    return ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--profile", action="store_true")  # ncu/nsys가 이 경로 실행
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  TOTAL_WEIGHTS={TOTAL}")
    assert TOTAL == 2819072, TOTAL  # challenge.py 레이아웃 일치 가드
    if args.check or not (args.bench or args.profile):
        _check(dev)
    if args.bench:
        if dev == "cpu": print("bench skip: GPU 없음")
        else: _bench(dev)
    if args.profile:
        x, w, cos, sin = _make_case(2048, dev)
        out = torch.empty(2048, D, device=dev)
        solve(x, out, w, cos, sin, 2048)  # 단일 호출 (프로파일러가 감쌈)
