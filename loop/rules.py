"""Hypothesis Engine (★내 코드) — 시드룰 매칭 → 병목 라벨 + 변형 프롬프트.

설계: design spec §3.2. 각 룰에 "왜 이 신호→이 병목" 근거 1줄 필수.
근거 없으면 룩업테이블, 있으면 perf 이해.

시드룰 = spec 표 그대로 (CUDAMaster 30% 임계값 + NVIDIA roofline 차용 근거).
이 표는 Rule Evolver(evolver.py)가 신뢰도를 갱신하는 '진화의 출발점'.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from signals import Signal

SLOW_LATENCY_US = 50.0  # "느림" 임계 (PoC FFN mul 76μs 기준 근방). evolver가 조정 가능.


@dataclass
class Rule:
    label: str                       # 병목 라벨
    cond: Callable[[Signal], bool]   # 발화 조건
    prompt: str                      # LLM에 줄 변형 가설 프롬프트
    rationale: str                   # 왜 이 신호→이 병목 (근거 1줄, 필수)
    priority: int                    # 낮을수록 먼저 (지배 병목 우선)
    # evolver가 갱신하는 진화 상태
    success: int = 0
    fail: int = 0
    retired: bool = False
    seed: bool = True                # 시드룰인가, evolver가 추가한 후보인가

    @property
    def confidence(self) -> float:
        """성공률. 시도 없으면 시드 신뢰 0.5 (탐색 여지)."""
        n = self.success + self.fail
        return 0.5 if n == 0 else self.success / n


def seed_rules() -> list[Rule]:
    """spec §3.2 시드 5~6개. 그대로 박음."""
    return [
        Rule(
            label="memory_saturated",
            cond=lambda t: t.bw_pct > 0.8,
            prompt="STOP: 대역폭 포화, 손댈 것 없음",
            rationale="elementwise는 BW가 천장 (NVIDIA roofline: AI<ridge → mem bound)",
            priority=0,
        ),
        Rule(
            label="uncoalesced",
            cond=lambda t: t.load_eff < 0.7,
            prompt="인덱싱 재배열 + shared 타일링으로 합착 접근 복구",
            rationale="비합착 접근 = 한 워프가 여러 캐시라인 → 대역폭 낭비",
            priority=1,
        ),
        Rule(
            label="reg_pressure",
            cond=lambda t: t.occ < 0.5 and t.reg > 64,
            prompt="스레드 수↓ 또는 __launch_bounds__로 레지스터 압박 완화",
            rationale="점유율 제한 = 레지스터 (SM당 레지스터 유한 → 동시 워프↓)",
            priority=2,
        ),
        Rule(
            label="oversync",
            cond=lambda t: t.stall_reason == "sync",
            prompt="__syncthreads 축소 / 더블버퍼링으로 동기화 stall 제거",
            rationale="과도한 동기화 = 워프가 배리어서 대기 = stall",
            priority=2,
        ),
        Rule(
            label="compute_bound",
            cond=lambda t: t.bw_pct < 0.6 and t.latency_us > SLOW_LATENCY_US,
            prompt="FMA 활용 / 정밀도 낮추기 (fp32→tf32) / 연산 융합",
            rationale="대역폭 여유인데 느림 = 연산이 천장 (roofline: AI>ridge)",
            priority=1,
        ),
        Rule(
            label="low_occupancy_latency",
            cond=lambda t: t.occ < 0.5 and t.reg <= 64,
            prompt="블록당 스레드↑ 또는 ILP↑로 지연 은닉 강화",
            rationale="점유율 낮지만 레지스터 아님 → 워프 부족, 지연 은닉 실패",
            priority=3,
        ),
    ]


@dataclass
class Hypothesis:
    label: str
    prompt: str
    rationale: str
    rule_idx: int           # 어느 룰이 발화했나 (ledger/evolver 추적용)
    is_stop: bool = False   # STOP 판정인가 (포화군)


def match(sig: Signal, rules: list[Rule]) -> Hypothesis | None:
    """가장 지배적 병목 1개 선택 (spec §3.2: 우선순위 → 신뢰도).

    탐색-활용: 같은 priority면 confidence 높은 룰 우선 (evolver 신뢰도 반영).
    retired 룰은 건너뜀.
    """
    live = [(i, r) for i, r in enumerate(rules) if not r.retired and r.cond(sig)]
    if not live:
        return None
    # priority 오름차순, 동률이면 confidence 내림차순
    i, r = min(live, key=lambda ir: (ir[1].priority, -ir[1].confidence))
    return Hypothesis(
        label=r.label, prompt=r.prompt, rationale=r.rationale,
        rule_idx=i, is_stop=r.label == "memory_saturated",
    )


if __name__ == "__main__":
    from signals import from_dict
    rules = seed_rules()

    # self-check 1: 포화 elementwise → STOP (spec sigmoid 예제)
    h = match(from_dict({"bw_pct": 0.85, "compute_tput": 0.15, "load_eff": 1.0}), rules)
    assert h is not None and h.label == "memory_saturated" and h.is_stop, h

    # self-check 2: 비합착 → uncoalesced
    h = match(from_dict({"bw_pct": 0.4, "load_eff": 0.55}), rules)
    assert h is not None and h.label == "uncoalesced", h

    # self-check 3: 연산바운드 (BW 여유 + 느림)
    h = match(from_dict({"bw_pct": 0.3, "latency_us": 76.0, "load_eff": 1.0}), rules)
    assert h is not None and h.label == "compute_bound", h

    # self-check 4: 아무 룰도 안 맞으면 None
    h = match(from_dict({"bw_pct": 0.5, "load_eff": 0.9, "occupancy": 0.9,
                         "latency_us": 10.0}), rules)
    assert h is None, h

    # self-check 5: priority 동률 시 confidence 높은 쪽 (탐색-활용)
    rules[1].success, rules[1].fail = 1, 9   # uncoalesced 신뢰 0.1
    rules[4].success, rules[4].fail = 9, 1   # compute_bound 신뢰 0.9 (priority 동일=1)
    h = match(from_dict({"bw_pct": 0.4, "load_eff": 0.55,
                         "latency_us": 76.0}), rules)  # 둘 다 발화 가능
    assert h is not None and h.label == "compute_bound", h  # 신뢰 높은 쪽

    print("rules.py self-check PASS")
