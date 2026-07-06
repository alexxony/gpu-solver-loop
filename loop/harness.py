"""Harness — 오케스트레이터. 한 라운드 = gen→gate→profile→hyp→ledger→evolve.

설계: design spec §2 아키텍처 다이어그램의 데이터 흐름.
글루(1·2·3)와 내 코드(4·5·6·7)를 엮는다. GPU 없이 FakeGlue로 전체 루프 검증.
수렴 정지(§3.4): N라운드 개선 없으면 종료 (무한 토큰 방지).
"""
from __future__ import annotations
from dataclasses import dataclass
from signals import from_dict, Context
from rules import seed_rules, match, Rule
from ledger import Ledger, RoundRecord
from evolver import evolve, EvolutionEvent
from glue import Generator, Gate, Profiler, code_hash

CONVERGE_PATIENCE = 6   # N라운드 연속 미개선 → 종료.
# 6 = RETIRE_AFTER_FAILS(3)+여유. retire 조건(fail>=3 & conf<=0.25 & n>=4) 채울
# 4R+ 확보해 수렴이 retire보다 먼저 터지는 것 방지 (T4 reg_pressure retire 관찰용).


@dataclass
class LoopResult:
    problem: str
    rounds: int
    best_metric: float
    stopped_reason: str          # "stop_label" | "converged" | "max_rounds" | "gate_fail"
    events: list[EvolutionEvent]


def _metric(sig, mode: str = "occupancy") -> float:
    """라운드 성과 지표. 항상 "높을수록 좋음" (best 비교 m>best 유지).

    mode="occupancy"(기본, 기존 동작): occupancy 우선, 없으면 bw_pct = 자원활용도.
    mode="latency": -latency_us 반환 = 더 빠른 커널일수록 큰 값(덜 음수). 성능 gain용.
      latency_us=0(미측정)이면 -inf 처리 안 함 → 0 반환(중립). 음수화로 m>best 로직 무변.
    """
    if mode == "latency":
        return -sig.latency_us if sig.latency_us > 0 else 0.0
    return sig.occupancy if sig.occupancy > 0 else sig.bw_pct


def run_loop(problem: str, glue, ledger: Ledger, rules: list[Rule] | None = None,
             max_rounds: int = 8, evolve_enabled: bool = True,
             metric_mode: str = "occupancy",
             ctx: Context | None = None) -> LoopResult:
    # ctx (design 07): 환경 가드. None=칩 미지=모든 가드 통과(종전 A100 동작 보존).
    #   T4 등 명시 시 match가 chip_cap 가드로 오탐 차단 → 진화가 흡수.
    # evolve_enabled=False = 정적 baseline (CUDAMaster류): 룰 고정, 진화 안 함.
    #   매치는 하되 success/fail 누적·retire 스킵 → 같은 룰 영원히 발화.
    # metric_mode="latency"면 _metric이 -latency_us(음수) 반환 → best 초기 -inf 필요
    #   (occupancy는 >=0이라 -1.0으로 첫 라운드 잡힘; latency 음수는 -1.0보다 작을 수 있음).
    rules = rules if rules is not None else seed_rules()
    all_events: list[EvolutionEvent] = []
    best = float("-inf") if metric_mode == "latency" else -1.0
    no_improve = 0
    prev_code = None

    for rnd in range(max_rounds):
        last = ledger.last(problem)
        hyp = None if last is None else _last_hyp(ledger, problem)
        gen = glue.generate(problem, hyp.prompt if hyp else None, prev_code)
        prev_code = gen.code

        gate = glue.check(gen.code, problem)
        if not gate.passed:
            # 틀린 커널 — 재생성 라운드 (가설 없이 다시). 기록만.
            ledger.append(RoundRecord(problem, rnd, gen.code_hash, {}, "gate_fail",
                                      "correctness 실패", -1, best, False, False))
            no_improve += 1
            if no_improve >= CONVERGE_PATIENCE:
                return LoopResult(problem, rnd + 1, best, "gate_fail", all_events)
            continue

        prof = glue.profile(gen.code, problem)
        sig = from_dict(prof.signal_dict)
        m = _metric(sig, metric_mode)
        improved = m > best
        if improved:
            best = m; no_improve = 0
        else:
            no_improve += 1

        # Hypothesis Engine: 다음 라운드용 가설 (현재 신호 + 환경 가드)
        h = match(sig, rules, ctx)
        rule_idx = h.rule_idx if h else -1

        rec = RoundRecord(
            problem, rnd, gen.code_hash, prof.signal_dict,
            h.label if h else "none", h.rationale if h else "", rule_idx,
            m, improved, True,
            note="STOP 판정" if (h and h.is_stop) else "",
        )
        ledger.append(rec)

        # Rule Evolver: 직전 라운드의 룰 결과로 진화 (이번 발화룰 평가)
        # evolve_enabled=False(정적 baseline)면 스킵 → 룰 영원히 고정 (CUDAMaster류).
        if evolve_enabled:
            ev = evolve(rules, ledger, rule_idx, improved)
            all_events.extend(ev)
            # 룰판이 바뀌면(retire/propose) "수렴" 가정 무효 — 새 룰로 다시 탐색해야.
            # no_improve 리셋해 converged 조기종료 막음 (폐기 후 발화 변경 관찰 가능).
            if any(e.kind in ("retire", "propose") for e in ev):
                no_improve = 0

        # STOP 라벨 = 포화군, 더 손댈 것 없음 (정직한 종료)
        if h and h.is_stop:
            return LoopResult(problem, rnd + 1, best, "stop_label", all_events)
        # 수렴 정지
        if no_improve >= CONVERGE_PATIENCE:
            return LoopResult(problem, rnd + 1, best, "converged", all_events)

    return LoopResult(problem, max_rounds, best, "max_rounds", all_events)


def _last_hyp(ledger: Ledger, problem: str):
    """직전 라운드의 가설 라벨/프롬프트 복원 (간이)."""
    last = ledger.last(problem)
    if last is None or last.rule_idx < 0:
        return None
    from types import SimpleNamespace
    return SimpleNamespace(prompt=last.hypothesis_label, label=last.hypothesis_label)


if __name__ == "__main__":
    import tempfile, os
    from glue import FakeGlue

    # ── 시나리오 A: 포화군 → 즉시 STOP ──
    fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd); os.unlink(p)
    try:
        glue = FakeGlue([({"bw_pct": 0.85, "load_eff": 1.0}, 10.0, True)])
        res = run_loop("sigmoid", glue, Ledger(p), max_rounds=8)
        assert res.stopped_reason == "stop_label", res.stopped_reason
        assert res.rounds == 1
        print("  scenario A (STOP 포화군): PASS")
    finally:
        os.path.exists(p) and os.unlink(p)

    # ── 시나리오 B: 최적화 가능군 → 메트릭 우상향 후 수렴 ──
    fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd); os.unlink(p)
    try:
        # occupancy 0.3→0.5→0.7 상승 후 정체 → converged.
        # 정체 구간 = CONVERGE_PATIENCE(6)만큼 필요. max_rounds 충분히 키움.
        plateau = ({"occupancy": 0.7, "bw_pct": 0.6, "load_eff": 0.7, "latency_us": 50}, 50, True)
        script = [
            ({"occupancy": 0.3, "bw_pct": 0.4, "load_eff": 0.55, "latency_us": 76}, 76, True),
            ({"occupancy": 0.5, "bw_pct": 0.5, "load_eff": 0.6, "latency_us": 60}, 60, True),
        ] + [plateau] * 7
        led = Ledger(p)
        res = run_loop("matmul", glue=FakeGlue(script), ledger=led, max_rounds=12)
        assert res.stopped_reason == "converged", res.stopped_reason
        curve = led.metric_curve("matmul")
        metrics = [m for _, m in curve]
        assert metrics[0] < metrics[2], "메트릭 우상향해야 (포폴 곡선)"
        assert res.best_metric == 0.7
        print(f"  scenario B (가능군 곡선): PASS — curve={metrics}")
    finally:
        os.path.exists(p) and os.unlink(p)

    # ── 시나리오 C: gate 연속 실패 → gate_fail 종료 ──
    fd, p = tempfile.mkstemp(suffix=".jsonl"); os.close(fd); os.unlink(p)
    try:
        glue = FakeGlue([({}, 0, False)] * 3)
        res = run_loop("broken", glue, Ledger(p), max_rounds=8)
        assert res.stopped_reason == "gate_fail", res.stopped_reason
        print("  scenario C (gate 실패): PASS")
    finally:
        os.path.exists(p) and os.unlink(p)

    print("harness.py self-check PASS")
