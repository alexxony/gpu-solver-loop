"""다문제 라운드 A — 룰 발화 관찰 (concept layer).

설계: GPU-Solver/docs/04-multiproblem-round-design.md §A.
3문제(llama/sigmoid/groupnorm)를 **공유 rules 객체** 1개로 각 1라운드 돌려,
문제별로 어느 룰이 발화하는지 ledger로 관찰. "신호가 다른 룰 깨우나?" 입증/반증.

⚠️ 이 실행은 concept layer만 — 신호→룰 매핑이 문제별로 다른지 확인. 룰 content
진화(promote/retire/propose)는 1라운드씩이라 거의 안 일어남(success/fail 미축적).
gain layer(진화 이득)는 B(LLM 다라운드)에서 별도 입증. 단일 라운드 결과로
차별점 입증 주장 금지.

전제: Colab watch 떠 있어야 (3왕복 = ncu 3회). mailbox/loop clone됨.
실행: python run_multiproblem.py
"""
from __future__ import annotations
import sys
from pathlib import Path

from runner import run_problem
from ledger import Ledger
from rules import seed_rules, Rule
from run_e2e import git_sync, MAILBOX, PROBLEMS   # git_sync·경로 재사용

PROBLEMS_LIST = ["llama", "sigmoid", "groupnorm"]
MP_LEDGER = MAILBOX.parent / "multiproblem-ledger.jsonl"


def main() -> int:
    if not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음: {MAILBOX}", file=sys.stderr)
        return 2

    # 누적 방지 — 이전 실행 ledger 제거 (깨끗한 발화 관찰).
    if MP_LEDGER.exists():
        MP_LEDGER.unlink()

    # ★ 핵심: 공유 rules 1개. 3문제 간 같은 객체 → 진화 누적 가능.
    rules: list[Rule] = seed_rules()

    print(f"다문제 A — 공유 rules({len(rules)}개), 문제={PROBLEMS_LIST}")
    print(f"  mailbox={MAILBOX} ledger={MP_LEDGER}")
    print("  ⚠️ concept layer만 (신호→룰). 진화 이득은 B에서.\n")

    for problem in PROBLEMS_LIST:
        solve_path = PROBLEMS / problem / "solve.py"
        if not solve_path.exists():
            print(f"  SKIP {problem}: solve.py 없음 ({solve_path})")
            continue
        seed = solve_path.read_text()
        print(f"── {problem} ── ({len(seed)} chars) REQ push → RES 대기...")
        res = run_problem(problem, seed, MAILBOX, MP_LEDGER,
                          sync_fn=git_sync, max_rounds=1,
                          poll_s=5.0, timeout_s=900.0, rules=rules)
        print(f"   rounds={res.rounds} stop={res.stopped_reason} "
              f"evol_events={len(res.events)}")

    # ── 발화 표 — 문제 → 발화 룰 라벨 (A의 핵심 산출) ──
    led = Ledger(str(MP_LEDGER))
    print("\n=== A 발화 표 (문제 → 발화 룰) ===")
    fired_labels = set()
    for problem in PROBLEMS_LIST:
        rec = led.last(problem)
        if rec is None:
            print(f"  {problem:10} → (라운드 없음)")
            continue
        stop = " [STOP]" if rec.note.startswith("STOP") else ""
        none = " ⚠️미발화(시드룰 빈곳)" if rec.rule_idx == -1 else ""
        print(f"  {problem:10} → {rec.hypothesis_label}{stop}{none} "
              f"(idx={rec.rule_idx}, passed={rec.passed})")
        if rec.rule_idx >= 0:
            fired_labels.add(rec.hypothesis_label)

    # ── A 판정 ──
    print(f"\n=== A 판정 ===")
    print(f"발화 라벨 종류: {len(fired_labels)} {sorted(fired_labels)}")
    if len(fired_labels) >= 2:
        print("✅ PASS — 신호가 ≥2종 다른 룰 깨움 = concept layer 실증. B 진행 의미 있음.")
    elif len(fired_labels) == 1:
        print("⚠️ 단일 라벨 — 신호 달라도 같은 룰. 룰 조건 재검토 후 B 판단.")
    else:
        print("⚠️ 전부 미발화 — 시드룰이 이 신호들 못 잡음. propose_candidate 동기 = B 재료.")
    print("\n⚠️ 경계: A PASS ≠ 차별점 입증. A = B가 의미있는 환경인지 게이트.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
