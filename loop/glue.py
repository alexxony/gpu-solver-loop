"""글루 (교체 가능) — Generator / Correctness Gate / Submit.

설계: design spec §2 컴포넌트1·2·3. "시키는 대로 코드만" = 내 지능 아님.
LLM API + GPU 실행 필요 → 진짜 구현은 Colab. 여기선 인터페이스(Protocol) +
GPU 없이 도는 FakeGlue (self-check/harness 테스트용).

Colab 붙을 때: RealGenerator(LLM API), RealGate(challenge.py reference_impl 비교),
RealProfiler(ncu) 로 교체. 인터페이스는 동일.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import hashlib


@dataclass
class GenResult:
    code: str
    code_hash: str


@dataclass
class GateResult:
    passed: bool
    max_abs_err: float = 0.0


@dataclass
class ProfileResult:
    signal_dict: dict       # 정규화된 Signal 필드 (signals.from_dict 입력)
    latency_us: float


class Generator(Protocol):
    def generate(self, problem: str, hypothesis_prompt: str | None,
                 prev_code: str | None) -> GenResult: ...


class Gate(Protocol):
    def check(self, code: str, problem: str) -> GateResult: ...


class Profiler(Protocol):
    def profile(self, code: str, problem: str) -> ProfileResult: ...


def code_hash(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


# ── GPU 없이 도는 가짜 글루 (결정론적 시나리오 재생) ──

class FakeGlue:
    """스크립트된 (signal, latency, pass) 시퀀스를 라운드마다 뱉음.

    harness/self-check가 GPU/LLM 없이 전체 루프를 돌려 ★4·5·6·7 검증.
    실제 perf 아님 — 진화 로직 통합 테스트용.
    """
    def __init__(self, script: list[tuple[dict, float, bool]]):
        self.script = script
        self.i = 0

    def generate(self, problem, hypothesis_prompt, prev_code) -> GenResult:
        code = f"# round {self.i} for {problem}\n# hyp: {hypothesis_prompt}\n"
        return GenResult(code, code_hash(code))

    def check(self, code, problem) -> GateResult:
        _, _, passed = self.script[self.i]
        return GateResult(passed, 0.0 if passed else 1.0)

    def profile(self, code, problem) -> ProfileResult:
        sig, lat, _ = self.script[self.i]
        self.i += 1
        return ProfileResult({**sig, "latency_us": lat}, lat)


if __name__ == "__main__":
    g = FakeGlue([({"bw_pct": 0.3}, 76.0, True), ({"bw_pct": 0.5}, 40.0, True)])
    r = g.generate("llama", "fuse silu+mul", None)
    assert r.code_hash and len(r.code_hash) == 10
    assert g.check(r.code, "llama").passed
    pr = g.profile(r.code, "llama")
    assert pr.signal_dict["bw_pct"] == 0.3 and pr.latency_us == 76.0
    assert g.i == 1  # profile이 커서 전진
    assert code_hash("a") != code_hash("b")
    print("glue.py self-check PASS")
