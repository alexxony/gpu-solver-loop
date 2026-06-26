"""gain 실험 1스텝 — 대행 generate로 진짜 GPU improved 변동 검증 (B).

설계: GPU-Solver/docs/04-multiproblem-round-design.md §B 본체.
CallbackGenerator(대행)로 sigmoid를 2라운드: R0=원본(base), R1=변형(가설 적용).
실제 Colab gate+ncu 측정 → signal 변동 → evolver가 improved 판정 → demote/promote.

⚠️ 이건 mechanism 실데이터 (대행 흐름 + 진짜 GPU improved 변동) 1점. gain layer
(다문제 평균 우위)는 3문제 다라운드 필요 — 이 1스텝은 "대행+GPU 닫힘 작동" 검증.

R1 변형 코드 = 외부 파일 인자(--variant). 대행(나)이 발화 룰 가설대로 만든 solve.py.

실행: python run_gain_round.py <problem> <variant_solve.py>
전제: Colab watch 살아있어야. mailbox/problems clone됨.
"""
from __future__ import annotations
import sys
from pathlib import Path

from runner import run_problem
from ledger import Ledger
from rules import seed_rules
from generator import CallbackGenerator
from run_e2e import git_sync, MAILBOX, PROBLEMS

GAIN_LEDGER = MAILBOX.parent / "gain-ledger.jsonl"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python run_gain_round.py <problem> <variant_solve.py>",
              file=sys.stderr)
        return 2
    problem = sys.argv[1]
    variant_path = Path(sys.argv[2])

    seed_path = PROBLEMS / problem / "solve.py"
    if not seed_path.exists():
        print(f"ERR: seed 없음 {seed_path}", file=sys.stderr); return 2
    if not variant_path.exists():
        print(f"ERR: variant 없음 {variant_path}", file=sys.stderr); return 2
    if not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음 {MAILBOX}", file=sys.stderr); return 2

    if GAIN_LEDGER.exists():
        GAIN_LEDGER.unlink()

    seed_code = seed_path.read_text()
    variant_code = variant_path.read_text()

    # 대행 콜백: R0(prev_code=None)는 seed 원본, R1+는 변형(가설 적용).
    # 발화 룰 가설을 보고 내(대행)가 만든 변형코드를 주입 = LLM generate 모방.
    calls = {"n": 0}
    def callback(prob, hyp, prev_code):
        i = calls["n"]; calls["n"] += 1
        print(f"  [generate R{i}] problem={prob} 가설={hyp!r}")
        if i == 0:
            return seed_code            # R0 = 원본 base
        return variant_code             # R1+ = 대행 변형 (가설 적용)

    gen = CallbackGenerator(callback)
    shared = seed_rules()

    print(f"gain 1스텝 — {problem}, 대행 generate (R0원본→R1변형)")
    print(f"  seed={seed_path.name} variant={variant_path.name}")
    print(f"  ⚠️ mechanism 실데이터 1점 (GPU improved 변동). gain layer는 3문제 다라운드.\n")

    res = run_problem(problem, seed_code, MAILBOX, GAIN_LEDGER,
                      sync_fn=git_sync, max_rounds=2, poll_s=5.0,
                      timeout_s=900.0, rules=shared, generator=gen,
                      evolve_enabled=True)

    led = Ledger(str(GAIN_LEDGER))
    recs = [r for r in led.records if r.problem == problem]
    print(f"\n=== 라운드 기록 ===")
    for r in recs:
        print(f"  R{r.round_idx}: {r.hypothesis_label} improved={r.improved} "
              f"passed={r.passed} bw_pct={r.signal.get('bw_pct')} "
              f"compute_tput={r.signal.get('compute_tput')}")

    print(f"\n=== 룰 진화 ===")
    for i, rr in enumerate(shared):
        n = rr.success + rr.fail
        if n > 0 or rr.retired:
            mark = " ★RETIRED" if rr.retired else ""
            print(f"  [{i}] {rr.label}: success={rr.success} fail={rr.fail} "
                  f"conf={rr.confidence:.2f}{mark}")
    print(f"  진화 이벤트: {[e.kind for e in res.events]}")
    print(f"  stop={res.stopped_reason}")

    print(f"\n⚠️ 경계: 이 데이터 = 대행+GPU 닫힘 작동 + improved 실변동. "
          f"gain layer(정적 대비 우위)는 3문제 다라운드.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
