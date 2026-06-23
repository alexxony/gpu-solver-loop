"""검증 실험: 정적 임계(CUDAMaster류) vs 진화 임계 — 다른 결정? 진화가 더 나은가?

GPU 불필요. 이번 세션 실측 ncu 분포(TF32 전/후) + R3~R7 결과로 시뮬레이션.
질문: 환경이 바뀔 때(TF32 적용 → 분포 재편) 정적 5% 게이트가 틀린 결정을 내리고
진화 게이트가 맞는 결정을 내리는가? = 차별점이 '우수'한지 측정.

실측 ground truth (이번 세션, 같은 하네스 median):
  커널          TF32전비중  TF32후비중  융합/최적화 실측 결과
  matmul         0.527      0.203      TF32로 -77% (큰 이득) ✅
  flash_attn     0.287      0.486      (이미 챔피언, 미시도)
  elementwise    0.077      0.129      융합=launch오버헤드 손해 (R7 예측) ❌
  mul (FFN)      0.065      0.110      R3' 융합 커널-39%지만 전체 1.01%(약), TF32후 승격시 8%회귀 ❌
  rmsnorm        0.028      0.046      (미시도)
"""
from __future__ import annotations
import statistics

# (이름, TF32전 비중, TF32후 비중, 실제 최적화하면 전체 개선됐나)
# improved_truth = 실측: 이 커널 타깃 최적화가 전체 latency 줄였나
KERNELS = [
    # name,        w_pre,  w_post, improved_truth (실측 결과)
    ("matmul",     0.527,  0.203,  True),    # TF32 큰 이득
    ("flash_attn", 0.287,  0.486,  False),   # 이미 챔피언, 추가 최적화 이득 못 냄(미시도→보수적 False)
    ("elementwise",0.077,  0.129,  False),   # 융합 손해 (R7)
    ("mul",        0.065,  0.110,  False),   # R3' 약적중+승격시 회귀
    ("rmsnorm",    0.028,  0.046,  False),   # 작음
]

STATIC_GATE = 0.05   # CUDAMaster류: 5% 한 번 정하고 고정


def otsu_threshold(values: list[float]) -> float:
    """1D Otsu: 두 그룹 분산 최소화하는 임계. ledger 비중값들에서 게이트 학습."""
    if len(values) < 2:
        return STATIC_GATE
    vs = sorted(values)
    best_t, best_var = vs[0], float("inf")
    for i in range(1, len(vs)):
        t = (vs[i - 1] + vs[i]) / 2
        lo = [v for v in vs if v <= t]
        hi = [v for v in vs if v > t]
        if not lo or not hi:
            continue
        # 그룹 내 분산 가중합 (Otsu = 클래스 내 분산 최소)
        wvar = (len(lo) * statistics.pvariance(lo) +
                len(hi) * statistics.pvariance(hi)) / len(vs)
        if wvar < best_var:
            best_var, best_t = wvar, t
    return best_t


def decisions(weights: dict, gate: float) -> dict:
    """게이트 적용: 각 커널 '시도할 가치(비중≥게이트)'인가."""
    return {k: (w >= gate) for k, w in weights.items()}


def evaluate(decision: dict, truth: dict) -> dict:
    """결정 vs 실측 정합. 시도했는데 개선=TP, 시도했는데 안됨=FP(헛수고),
    안시도했는데 개선가능=FN(놓침), 안시도&안됨=TN(올바른 스킵)."""
    tp = fp = fn = tn = 0
    for k in decision:
        d, t = decision[k], truth[k]
        if d and t: tp += 1
        elif d and not t: fp += 1
        elif not d and t: fn += 1
        else: tn += 1
    # 헛수고(FP) + 놓침(FN) = 나쁜 결정. 적을수록 좋음.
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "bad": fp + fn}


if __name__ == "__main__":
    truth = {n: imp for n, w_pre, w_post, imp in KERNELS}
    w_pre = {n: w_pre for n, w_pre, w_post, imp in KERNELS}
    w_post = {n: w_post for n, w_pre, w_post, imp in KERNELS}

    print("=== 검증: 정적 게이트(5% 고정) vs 진화 게이트(분포 Otsu) ===\n")

    # TF32 후 분포 = 환경 바뀐 시점. 이때 두 게이트가 다른 결정?
    static_dec = decisions(w_post, STATIC_GATE)
    evolved_gate = otsu_threshold(list(w_post.values()))
    evolved_dec = decisions(w_post, evolved_gate)

    print(f"TF32 후 분포: {w_post}")
    print(f"정적 게이트 = {STATIC_GATE:.3f} (고정)")
    print(f"진화 게이트 = {evolved_gate:.3f} (Otsu 학습)\n")

    print(f"{'커널':12} {'비중후':>7} {'실측개선':>8} {'정적결정':>8} {'진화결정':>8}")
    for n in truth:
        print(f"{n:12} {w_post[n]:7.3f} {str(truth[n]):>8} "
              f"{str(static_dec[n]):>8} {str(evolved_dec[n]):>8}")

    s_eval = evaluate(static_dec, truth)
    e_eval = evaluate(evolved_dec, truth)
    print(f"\n정적 게이트: 헛수고(FP)={s_eval['fp']} 놓침(FN)={s_eval['fn']} → 나쁜결정 {s_eval['bad']}")
    print(f"진화 게이트: 헛수고(FP)={e_eval['fp']} 놓침(FN)={e_eval['fn']} → 나쁜결정 {e_eval['bad']}")

    verdict = ("진화 우수" if e_eval['bad'] < s_eval['bad']
               else "동률 (차별점 코드는 있으나 이 케이스선 이득 무)" if e_eval['bad'] == s_eval['bad']
               else "정적이 나음 (진화 역효과)")
    print(f"\n판정: {verdict}")

    # 정직성 가드: 표본 작으면 명시
    print(f"\n[표본 한계] 커널 {len(KERNELS)}개 = Otsu 통계 빈약. "
          f"진짜 검증은 수십 라운드 누적 필요. 이건 '방향' 점검.")
