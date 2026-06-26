"""옵션 1 — 진화 ON/OFF 비교 (LLM·GPU 0). evolver demote/retire 메커니즘 검증.

설계: GPU-Solver/docs/04-multiproblem-round-design.md §B 옵션1.
A에서 발견: fp32_no_tensorcore가 sigmoid(메모리바운드) 오발화. 이 룰이 측정 피드백
(improved=False 반복)으로 demote→retire 되는지 = 차별점 메커니즘 실증.

fake responder = sigmoid 실측 신호(A의 RES, bw 0.668 등) 고정 주입. GPU 왕복 0.
같은 신호 반복 → fp32 룰 매 라운드 발화하나 metric 정체 → improved=False →
evolver demote 누적 → 4회 fail시 retire.

비교:
  OFF(정적) = 매 라운드 seed_rules() 재생성 → fp32 룰 영원히 발화 (CUDAMaster류).
  ON(진화)  = 공유 rules → fp32 반복 fail → retire → 폐기 후 발화 바뀜.

⚠️ 이건 evolver 메커니즘이 실측 fail로 도는지 검증 (concept→mechanism). harness가
가설→코드변형 닫힘은 안 함(FixedGenerator) → 자동 코드변형은 B(RealGenerator).
gain layer(정적 대비 측정 이득)는 RealGenerator 다라운드서 별도 입증.

실행: python run_evolution_compare.py
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

from runner import run_problem
from ledger import Ledger
from rules import seed_rules
from mailbox import fake_colab_respond

# A에서 실측한 sigmoid 신호 (RES-2d366 / multiproblem-ledger). 메모리바운드.
# fp32_no_tensorcore 조건(weight≥0.05 + compute_tput>0 + not tensorcore) 충족 →
# 이 룰이 발화하나, sigmoid엔 matmul 없어 TF32 가설 무효 = 개선 0 = demote 대상.
SIGMOID_SIGNAL = {
    "occupancy": 0.45, "reg_per_thread": 32, "l2_hit": 0.95, "load_eff": 0.9,
    "bw_pct": 0.668, "compute_tput": 0.696, "latency_us": 120.0,
    "weight_pct": 1.0, "tensorcore_active": False,
}


def _responder(cmd: dict) -> dict:
    """가짜 Colab — sigmoid 신호 고정 반환 (코드 무관, metric 정체 유도)."""
    return {"passed": True, "max_abs_err": 1e-7,
            "signal_dict": dict(SIGMOID_SIGNAL),
            "latency_us": SIGMOID_SIGNAL["latency_us"], "error": None}


def _run(rounds: int, shared_rules, evolve_enabled: bool):
    """sigmoid를 rounds회 돌림.

    evolve_enabled=False = 정적 baseline (CUDAMaster류): 룰 고정, 진화 스킵.
    True = 진화 ON, shared_rules 객체로 success/fail 누적·retire.
    """
    with tempfile.TemporaryDirectory() as d:
        def sync_fn(_mb):
            fake_colab_respond(d, _responder)

        led_path = Path(d) / "ledger.jsonl"
        res = run_problem("sigmoid", "# sigmoid seed (fixed)", d, led_path,
                          sync_fn=sync_fn, max_rounds=rounds,
                          poll_s=0.0, timeout_s=10.0, rules=shared_rules,
                          evolve_enabled=evolve_enabled)
        led = Ledger(str(led_path))
        fired = [(r.round_idx, r.hypothesis_label, r.rule_idx, r.improved)
                 for r in led.records if r.problem == "sigmoid"]
        return res, fired


def main() -> int:
    ROUNDS = 6
    print(f"진화 ON/OFF 비교 — sigmoid {ROUNDS}라운드, 신호 고정(bw={SIGMOID_SIGNAL['bw_pct']})")
    print("  fp32_no_tensorcore 오발화가 측정 피드백으로 폐기되는지 검증\n")

    # ── OFF: 정적 baseline (매 라운드 새 seed_rules) ──
    print("=== 진화 OFF (정적 — CUDAMaster류) ===")
    res_off, fired_off = _run(ROUNDS, shared_rules=None, evolve_enabled=False)
    for rnd, label, idx, imp in fired_off:
        print(f"  R{rnd}: {label} (idx={idx}, improved={imp})")
    off_labels = [f[1] for f in fired_off]
    print(f"  → stop={res_off.stopped_reason}, 발화 변천: {off_labels}")

    # ── ON: 공유 rules + evolve ──
    print("\n=== 진화 ON (공유 rules + evolver) ===")
    shared = seed_rules()
    res_on, fired_on = _run(ROUNDS, shared_rules=shared, evolve_enabled=True)
    for rnd, label, idx, imp in fired_on:
        print(f"  R{rnd}: {label} (idx={idx}, improved={imp})")
    on_labels = [f[1] for f in fired_on]
    print(f"  → stop={res_on.stopped_reason}, 발화 변천: {on_labels}")

    # 진화 상태 — fp32 룰(idx 1) 운명
    print("\n=== 룰 진화 결과 (ON, 공유 rules) ===")
    for i, r in enumerate(shared):
        n = r.success + r.fail
        mark = " ★RETIRED" if r.retired else ""
        if n > 0 or r.retired:
            print(f"  [{i}] {r.label}: success={r.success} fail={r.fail} "
                  f"conf={r.confidence:.2f}{mark}")
    evol_kinds = [e.kind for e in res_on.events]
    print(f"  진화 이벤트: {evol_kinds}")

    # ── 판정 ──
    print("\n=== 판정 ===")
    fp32_retired = any(r.label == "fp32_no_tensorcore" and r.retired for r in shared)
    off_stuck = off_labels.count("fp32_no_tensorcore") > on_labels.count("fp32_no_tensorcore")
    if fp32_retired:
        print("✅ ON: fp32_no_tensorcore 오발화 룰 RETIRE = 측정 피드백으로 폐기.")
        print("   OFF(정적)는 같은 오발화 영원 반복 → 진화가 헛가설 차단 = 메커니즘 실증.")
    elif "retire" in evol_kinds or "demote" in evol_kinds:
        print("⚠️ demote는 일어났으나 retire 미달 (라운드↑ 또는 RETIRE 조건 확인).")
        print(f"   ON 발화 변천이 OFF와 다른지: {on_labels != off_labels}")
    else:
        print("❌ 진화 미발생 — demote/retire 안 일어남. evolve 흐름 점검 필요.")
    print("\n⚠️ 경계: 메커니즘 검증(demote/retire가 실측 fail로 돎). gain layer(정적 대비")
    print("   측정 이득)는 RealGenerator 다라운드 + 3문제 GT서 별도 입증 (B 다음 단계).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
