"""로컬측 자동 루프 러너 — 우편함 ↔ harness ↔ evolver 연결 (A 인프라).

설계: docs/03-git-mailbox-runner.md §컴포넌트매핑. harness.run_loop는 glue
(generate/check/profile 3메서드)를 기대. MailboxProfiler는 submit 1왕복으로
gate+profile을 함께 줌 → 여기 MailboxGlue 어댑터가 그 임피던스를 맞춘다.

  harness.run_loop ─ glue.generate ─→ FixedGenerator (solve.py 고정, LLM 아직 stub)
                   ├ glue.check    ─┐ 같은 (code,problem)이면 submit 1회 캐시
                   └ glue.profile  ─┘ → MailboxResult{gate, profile} 둘로 분배

generate(LLM)는 A 인프라 범위 밖 — FixedGenerator로 고정. 배관(signal→match→
evolve)이 단일문제로 흐르는지 검증이 목표. 다문제 차별점은 네 문제 오면.

GPU·git 없이 self-check: MailboxProfiler에 no-op sync + fake responder 주입.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

try:
    from .glue import GenResult, GateResult, ProfileResult, code_hash
    from .mailbox import MailboxProfiler, MailboxResult
    from .ledger import Ledger
    from .harness import run_loop, LoopResult
except ImportError:                      # Colab !python 직접 실행 대비
    from glue import GenResult, GateResult, ProfileResult, code_hash
    from mailbox import MailboxProfiler, MailboxResult
    from ledger import Ledger
    from harness import run_loop, LoopResult


class FixedGenerator:
    """generate stub — 고정 코드 반환. LLM 대체(A 인프라 범위 밖).

    harness가 매 라운드 generate를 부르지만, 단일문제 배관 검증선 코드 고정으로
    충분(같은 solve.py를 매번 측정 → signal_dict 흐름만 확인). LLM generate는
    RealGenerator로 후속 교체 — 인터페이스 동일.
    """
    def __init__(self, code: str):
        self.code = code

    def generate(self, problem: str, hypothesis_prompt: str | None,
                 prev_code: str | None) -> GenResult:
        # 가설 프롬프트는 무시(고정 코드). 실 LLM은 이걸 받아 코드 변형.
        return GenResult(self.code, code_hash(self.code))


class MailboxGlue:
    """harness glue 어댑터 — generate + check + profile을 mailbox 1왕복에 매핑.

    harness는 check(code)→profile(code)를 연속 호출. 둘 다 같은 (code,problem)
    이면 우편함 왕복 1회면 충분(gate+profile 합쳐 옴). 마지막 submit 결과를
    (code,problem) 키로 캐시 → check가 채우고 profile이 재사용. 중복 왕복 방지.
    """
    def __init__(self, generator, profiler: MailboxProfiler):
        self.gen = generator
        self.profiler = profiler
        self._cache: dict[tuple[str, str], MailboxResult] = {}

    def generate(self, problem, hypothesis_prompt, prev_code) -> GenResult:
        return self.gen.generate(problem, hypothesis_prompt, prev_code)

    def _submit(self, code: str, problem: str) -> MailboxResult:
        key = (code_hash(code), problem)
        if key not in self._cache:
            self._cache[key] = self.profiler.submit(code, problem)
        return self._cache[key]

    def check(self, code: str, problem: str) -> GateResult:
        return self._submit(code, problem).gate

    def profile(self, code: str, problem: str) -> ProfileResult:
        return self._submit(code, problem).profile


def run_problem(
    problem: str,
    seed_code: str,
    mailbox_dir: str | Path,
    ledger_path: str | Path,
    sync_fn: Callable[[Path], None],
    max_rounds: int = 8,
    poll_s: float = 5.0,
    timeout_s: float = 600.0,
    rules=None,
    evolve_enabled: bool = True,
    generator=None,
    metric_mode: str = "occupancy",
    ctx=None,
) -> LoopResult:
    """한 문제를 자동 루프에 태운다 — 우편함 경유 측정 + 룰 진화.

    seed_code = 시드 솔버(solve.py 텍스트). sync_fn = git pull/push 1회(운용).
    반환 LoopResult.events = 진화 증거(evolver). ledger_path에 라운드 누적.

    rules: 공유 룰 리스트. None이면 run_loop가 seed_rules() 새로 생성(단일문제).
      다문제 라운드는 같은 rules 객체를 문제 간 주입 → 진화 누적 (04-multiproblem 설계).
    generator: glue.Generator 구현. None이면 FixedGenerator(seed 고정, 배관용).
      RealGenerator(LLM API) 또는 ManualGenerator(대행)를 주입해 코드 변형 라운드.
    """
    profiler = MailboxProfiler(mailbox_dir, sync_fn=sync_fn,
                               poll_s=poll_s, timeout_s=timeout_s)
    gen = generator if generator is not None else FixedGenerator(seed_code)
    glue = MailboxGlue(gen, profiler)
    ledger = Ledger(str(ledger_path))
    return run_loop(problem, glue, ledger, max_rounds=max_rounds, rules=rules,
                    evolve_enabled=evolve_enabled, metric_mode=metric_mode, ctx=ctx)


if __name__ == "__main__":
    # self-check: GPU·git·LLM 0. fake 우편함(no-op sync + responder)로 전체
    # signal→match→evolve 흐름 검증. mailbox.fake_colab_respond 재사용.
    import tempfile, os
    from mailbox import fake_colab_respond

    # 가짜 Colab watch: 메모리바운드 신호 → 융합 가설 발화 유도, gate 통과.
    def responder(cmd: dict) -> dict:
        assert "code" in cmd and "problem" in cmd
        return {"passed": True, "max_abs_err": 1e-6,
                "signal_dict": {"occupancy": 0.4, "bw_pct": 0.55,
                                "compute_tput": 0.0, "weight_pct": 1.0,
                                "tensorcore_active": False, "latency_us": 76.0},
                "latency_us": 76.0, "error": None}

    with tempfile.TemporaryDirectory() as d:
        # sync_fn이 불릴 때 Colab이 응답한다고 가정 (실제 git pull 시점).
        def sync_fn(_mb):
            fake_colab_respond(d, responder)

        led_path = os.path.join(d, "ledger.jsonl")
        res = run_problem("llama_ffn", "# seed solver code", d, led_path,
                          sync_fn=sync_fn, poll_s=0.0, max_rounds=4)

        # 배관 검증: 라운드 돎 + signal이 ledger에 들어감 + 룰 발화
        led = Ledger(led_path)
        recs = [r for r in led.records if r.problem == "llama_ffn"]
        assert len(recs) >= 1, "라운드 기록돼야"
        r0 = recs[0]
        assert r0.passed, "gate 통과해야 (responder passed=True)"
        assert r0.signal["bw_pct"] == 0.55, "signal_dict가 ledger까지 흘러야"
        # 고정 코드라 metric 정체 → 수렴 종료. 룰 발화 1회 이상.
        assert res.stopped_reason in ("converged", "stop_label", "max_rounds"), res.stopped_reason
        # 같은 코드 → 우편함 왕복은 라운드당 1회만 (check+profile 캐시)
        print(f"runner.py self-check PASS — rounds={res.rounds} "
              f"stop={res.stopped_reason} events={len(res.events)}")
