"""Rule Evolver (★★유일 차별점★★) — Ledger 이력으로 룰 자체를 진화.

설계: design spec §3.3. 6개 선행(CUDAMaster·KernelAgent·SOLAR·Nsight·
DSL+SOL·CudaForge) 전부 정적/일회캘리브 임계값 — 룰이 결과 피드백으로
갱신되는 시스템 = 없음. 이 하나가 빈 곳.

CUDAMaster: Otsu 일회 캘리브 후 30% 고정.
우리: 라운드 결과로 룰 신뢰도·임계값 갱신/폐기/후보추가.

면접 한 마디: "골격은 CUDAMaster가 풀었다. 인정하고 차용했다.
내 기여는 그 룰표가 측정 피드백으로 진화한다는 것."
"""
from __future__ import annotations
from dataclasses import dataclass
from rules import Rule
from ledger import Ledger

RETIRE_AFTER_FAILS = 3       # 연속/누적 실패 N회 → 폐기 후보
RETIRE_MIN_CONF = 0.25       # 신뢰도 이 밑 + 충분히 시도 → 폐기


@dataclass
class EvolutionEvent:
    """진화 1건. before/after 증거 (성공기준 b: 메타루프 증거 1장)."""
    rule_idx: int
    label: str
    kind: str                # "promote" | "demote" | "retire" | "propose"
    conf_before: float
    conf_after: float
    reason: str


def update_from_round(rules: list[Rule], rule_idx: int, improved: bool
                      ) -> EvolutionEvent:
    """한 라운드 결과 → 발화한 룰의 성공/실패 카운트 갱신.

    improved=True → 이 룰의 가설이 메트릭 올림 = 성공.
    """
    r = rules[rule_idx]
    before = r.confidence
    if improved:
        r.success += 1
        kind, reason = "promote", "가설 적용 후 메트릭 개선"
    else:
        r.fail += 1
        kind, reason = "demote", "가설 적용해도 개선 없음"
    return EvolutionEvent(rule_idx, r.label, kind, before, r.confidence, reason)


def retire_pass(rules: list[Rule]) -> list[EvolutionEvent]:
    """반복 실패 룰 폐기 (spec §3.3 폐기 후보, §3.4 수렴).

    틀린 정적 임계값이 측정에 의해 강등→폐기되는 것 = CUDAMaster와의 격차.
    """
    events = []
    for i, r in enumerate(rules):
        if r.retired:
            continue
        n = r.success + r.fail
        # conf<=RETIRE_MIN_CONF (경계 포함): demote 3회면 정확히 0.25 도달 →
        # `<`면 한 끗 미달로 영영 폐기 안 됨 (run_evolution_compare서 발견한 버그).
        if r.fail >= RETIRE_AFTER_FAILS and r.confidence <= RETIRE_MIN_CONF and n >= 4:
            before = r.confidence
            r.retired = True
            events.append(EvolutionEvent(
                i, r.label, "retire", before, before,
                f"{r.fail}회 실패, 신뢰도 {before:.2f} — 정적 임계값이 틀렸다고 측정이 판정",
            ))
    return events


def propose_candidate(rules: list[Rule], led: Ledger) -> EvolutionEvent | None:
    """2차: 새 패턴 후보 제안 (spec §3.3 새 패턴 후보 제안).

    PoC 신호: 어떤 룰도 발화 못 했는데(매칭 None) 개선이 일어난 라운드가
    반복되면 = 시드룰이 못 잡는 패턴 존재 → 후보 룰 1개 추가.
    MVP에선 placeholder: 빈 매칭+개선 2회 이상이면 stub 후보 등록.
    """
    unexplained = [r for r in led.records if r.rule_idx == -1 and r.improved]
    if len(unexplained) < 2:
        return None
    if any(r.seed is False and r.label == "discovered_pattern" for r in rules):
        return None  # 이미 추가됨
    cand = Rule(
        label="discovered_pattern",
        cond=lambda t: False,   # 조건 미정 — 사람이 채울 후보 자리 (정직)
        prompt="(후보) Ledger가 발견한 미설명 개선 패턴 — 조건 수동 정의 필요",
        rationale="시드룰 미발화 라운드서 반복 개선 → 빈 곳 존재 신호",
        priority=5, seed=False,
    )
    rules.append(cand)
    return EvolutionEvent(len(rules) - 1, cand.label, "propose", 0.5, 0.5,
                          f"{len(unexplained)}회 미설명 개선 → 후보 룰 추가")


def evolve(rules: list[Rule], led: Ledger, rule_idx: int, improved: bool
           ) -> list[EvolutionEvent]:
    """한 라운드 후 전체 진화 패스: 갱신 → 폐기 → 후보제안."""
    events = []
    if rule_idx >= 0:
        events.append(update_from_round(rules, rule_idx, improved))
    events.extend(retire_pass(rules))
    cand = propose_candidate(rules, led)
    if cand:
        events.append(cand)
    return events


if __name__ == "__main__":
    import tempfile, os
    from rules import seed_rules
    from ledger import RoundRecord

    # ── 차별점 증명 시나리오: 틀린 정적 임계값이 측정으로 강등→폐기 ──
    # 가상: compute_bound 룰(idx 4)이 어떤 문제서 계속 틀림.
    #       CUDAMaster라면 30% 고정 → 영원히 같은 틀린 가설.
    #       우리 evolver는 4연속 실패 후 폐기.
    rules = seed_rules()
    fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd); os.unlink(p)
    try:
        led = Ledger(p)
        conf0 = rules[4].confidence
        assert conf0 == 0.5  # 시드 초기 신뢰

        # 4라운드 연속 실패 주입
        for i in range(4):
            rec = RoundRecord("badprob", i, f"h{i}", {"bw_pct": 0.3},
                              "compute_bound", "BW여유느림", 4, 0.3, False, True)
            led.append(rec)
            evolve(rules, led, rule_idx=4, improved=False)

        assert rules[4].fail == 4, rules[4].fail
        assert rules[4].retired, "4연속 실패 룰이 폐기돼야 함 (차별점)"
        assert rules[4].confidence < conf0, "신뢰도 하락해야 (before>after)"

        # 폐기된 룰은 이후 match에서 제외 (rules.match가 retired 건너뜀)
        from rules import match
        from signals import from_dict
        h = match(from_dict({"bw_pct": 0.3, "latency_us": 76.0, "load_eff": 1.0}),
                  rules)
        assert h is None or h.rule_idx != 4, "폐기 룰은 재발화 안 됨"

        # ── promote 시나리오: 맞는 룰은 신뢰도 상승 ──
        rules2 = seed_rules()
        led2 = Ledger(p + "2")
        for i in range(3):
            evolve(rules2, led2, rule_idx=1, improved=True)  # uncoalesced 연속 성공
        assert rules2[1].confidence > 0.5, "성공 룰 신뢰도 상승해야"
        assert rules2[1].success == 3

        # ── 후보 제안 시나리오: 미설명 개선 2회 → 새 룰 ──
        rules3 = seed_rules()
        led3 = Ledger(p + "3")
        n_before = len(rules3)
        for i in range(2):
            led3.append(RoundRecord("x", i, "h", {}, "none", "", -1, 0.5, True, True))
            evolve(rules3, led3, rule_idx=-1, improved=True)
        assert len(rules3) == n_before + 1, "후보 룰 추가돼야"
        assert rules3[-1].label == "discovered_pattern" and not rules3[-1].seed

        print("evolver.py self-check PASS — 차별점(룰 진화) 검증됨")
    finally:
        for q in (p, p + "2", p + "3"):
            if os.path.exists(q):
                os.unlink(q)
