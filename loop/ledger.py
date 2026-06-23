"""Run Ledger — 매 라운드 (코드/메트릭/가설/근거/결과) 적재. 곡선+로그 원천.

설계: design spec §2 컴포넌트7, §3.0 입력4(이전 라운드 로그).
JSONL 1줄=1라운드. Rule Evolver가 이 이력을 읽어 룰 신뢰도를 갱신.
포폴 결과물(percentile 상승 곡선 + 진화 로그)이 여기서 나온다.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class RoundRecord:
    problem: str
    round_idx: int
    code_hash: str               # solve.py 버전 식별 (긴 코드 대신 해시)
    signal: dict                 # 정규화 Signal (Trace Parser 출력)
    hypothesis_label: str        # 발화한 병목 라벨
    hypothesis_rationale: str
    rule_idx: int                # 어느 룰이 발화 (evolver 추적 키)
    metric: float                # 이번 라운드 성과 지표 (occupancy 또는 bw_pct 등)
    improved: bool               # 직전 대비 개선됐나 (evolver 성공/실패 판정)
    passed: bool                 # correctness gate 통과했나
    note: str = ""               # "flash 꺼짐 발견" 같은 반증 메모


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: list[RoundRecord] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.records.append(RoundRecord(**json.loads(line)))

    def append(self, rec: RoundRecord) -> None:
        self.records.append(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def last(self, problem: str) -> RoundRecord | None:
        for r in reversed(self.records):
            if r.problem == problem:
                return r
        return None

    def history_for_rule(self, rule_idx: int) -> list[RoundRecord]:
        """특정 룰이 발화한 모든 라운드 (evolver가 신뢰도 갱신할 때 봄)."""
        return [r for r in self.records if r.rule_idx == rule_idx]

    def metric_curve(self, problem: str) -> list[tuple[int, float]]:
        """(round_idx, metric) — 포폴 곡선 원천."""
        return [(r.round_idx, r.metric) for r in self.records if r.problem == problem]

    def tried_rules(self, problem: str) -> set[int]:
        """이 문제서 이미 실패한 룰 (spec §3.4: 같은 문제 재시도 안 함)."""
        return {r.rule_idx for r in self.records
                if r.problem == problem and not r.improved}


if __name__ == "__main__":
    import tempfile, os
    fd, p = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(p)  # Ledger가 새로 만들게
    try:
        led = Ledger(p)
        led.append(RoundRecord("llama", 0, "abc", {"bw_pct": 0.3}, "compute_bound",
                               "BW 여유 느림", 4, 0.30, False, True))
        led.append(RoundRecord("llama", 1, "def", {"bw_pct": 0.5}, "compute_bound",
                               "BW 여유 느림", 4, 0.55, True, True, "fused silu+mul"))
        # 재로드 (디스크 라운드트립)
        led2 = Ledger(p)
        assert len(led2.records) == 2
        assert led2.last("llama").round_idx == 1
        assert led2.metric_curve("llama") == [(0, 0.30), (1, 0.55)]
        assert led2.history_for_rule(4) and len(led2.history_for_rule(4)) == 2
        assert led2.tried_rules("llama") == {4}  # round0 미개선 → 실패 기록
        print("ledger.py self-check PASS")
    finally:
        if os.path.exists(p):
            os.unlink(p)
