"""RealGenerator — LLM(Claude API)이 가설대로 solve.py 변형 (B 본체).

설계: GPU-Solver/docs/04-multiproblem-round-design.md §B 본체.
glue.Generator 계약: generate(problem, hypothesis_prompt, prev_code) → GenResult.
FixedGenerator(고정 코드) 대체 — LLM이 hypothesis_prompt 받아 prev_code 변형 →
진짜 improved 변동 → evolver promote/demote/retire가 측정 피드백으로 작동 = gain layer.

generate = 순수 API 호출(GPU 불요) → 로컬서 실행. 변형 코드를 REQ로 Colab 보내
gate+ncu(A100). 우편함 프로토콜 무변 — code 필드에 변형 코드 넣기만.

call_fn 주입 = API 호출 추상화. self-check는 fake call_fn으로 SDK·키·네트워크 0.
실 호출 = anthropic SDK lazy import (없으면 명확한 에러).
"""
from __future__ import annotations
import re
from typing import Callable

try:
    from .glue import GenResult, code_hash
except ImportError:
    from glue import GenResult, code_hash

MODEL = "claude-opus-4-8"
MAX_TOKENS = 8000

SYSTEM_PROMPT = """\
너는 GPU 커널 최적화 전문가다. SystemVerilog가 아니라 PyTorch/Triton solve.py를 다룬다.

주어진 solve.py를 '최적화 가설'대로 변형해 더 빠른 버전을 만든다.

절대 규칙 (어기면 gate 실패 = 라운드 무효):
1. executor 어댑터 계약 5심볼 유지: make_case(size,device)/run_solve(case,device)/
   reference(case,device)/GATE_SIZES/PROFILE_SIZE. 시그니처·반환형 절대 변경 금지.
2. reference()는 정답 기준 — 절대 손대지 마라. solve 커널만 바꾼다.
3. 출력은 완전한 solve.py 1개. ```python 코드블록 1개로만 답한다. 설명 금지.
4. 수치 정확도 유지 (gate가 reference와 atol 비교). 정확도 깨는 변형 금지.
"""


def _build_user_msg(problem: str, hypothesis_prompt: str | None,
                    prev_code: str) -> str:
    hyp = hypothesis_prompt or "특별한 가설 없음 — 일반적 최적화(메모리 접근/융합/점유율) 시도."
    return (
        f"문제: {problem}\n\n"
        f"최적화 가설:\n{hyp}\n\n"
        f"현재 solve.py:\n```python\n{prev_code}\n```\n\n"
        f"위 가설대로 solve 커널을 변형한 완전한 solve.py를 코드블록 1개로 답하라."
    )


def _extract_code(text: str) -> str:
    """LLM 응답서 ```python ... ``` 블록 추출. 없으면 전체 텍스트 (fallback)."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _anthropic_call(system: str, user: str) -> str:
    """실 Claude API 호출. anthropic SDK + ANTHROPIC_API_KEY 필요 (lazy)."""
    import anthropic                      # 없으면 ImportError → 명확한 신호
    client = anthropic.Anthropic()       # ANTHROPIC_API_KEY 환경변수 자동 사용
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


class RealGenerator:
    """LLM이 가설대로 solve.py 변형. call_fn 주입(테스트), 기본 = Claude API.

    seed_code = 1라운드 시작 코드(현 챔피언 solve.py). 이후 라운드는 prev_code
    (직전 라운드 코드) 변형. hypothesis_prompt = harness가 준 발화룰 가설.
    """
    def __init__(self, seed_code: str,
                 call_fn: Callable[[str, str], str] = _anthropic_call):
        self.seed = seed_code
        self.call = call_fn

    def generate(self, problem: str, hypothesis_prompt: str | None,
                 prev_code: str | None) -> GenResult:
        base = prev_code or self.seed     # 1라운드는 seed, 이후 직전 코드
        user = _build_user_msg(problem, hypothesis_prompt, base)
        raw = self.call(SYSTEM_PROMPT, user)
        code = _extract_code(raw)
        if not code:                      # 빈 응답 → seed 폴백 (라운드 안 죽임)
            code = base
        return GenResult(code, code_hash(code))


if __name__ == "__main__":
    # self-check: API·SDK·키·네트워크 0. fake call_fn으로 generate 흐름 검증.

    SEED = "def solve(X, Y, N):\n    pass  # seed kernel\n"

    # 1. 정상 — 코드블록 추출 + 가설/prev_code가 프롬프트에 들어감
    seen = {}
    def fake_call(system, user):
        seen["system"], seen["user"] = system, user
        return "변형했다:\n```python\ndef solve(X, Y, N):\n    return fast(X, Y, N)\n```\n끝."
    g = RealGenerator(SEED, call_fn=fake_call)
    r = g.generate("sigmoid", "메모리 융합 시도", prev_code=None)
    assert "def solve" in r.code and "fast" in r.code, r.code
    assert "설명" not in r.code and "변형했다" not in r.code, "코드블록만 추출해야"
    assert len(r.code_hash) == 10
    assert "메모리 융합 시도" in seen["user"], "가설이 프롬프트에 들어가야"
    assert SEED.strip() in seen["user"], "1라운드는 seed가 base"
    assert "어댑터 계약" in seen["system"], "계약 보존 지시"

    # 2. prev_code 있으면 그게 base (seed 아님)
    PREV = "def solve(X, Y, N):\n    return prev_version()\n"
    g.generate("sigmoid", "더 최적화", prev_code=PREV)
    assert "prev_version" in seen["user"] and SEED.strip() not in seen["user"]

    # 3. 코드블록 없는 응답 → 전체 텍스트 fallback (빈 건 아님)
    g2 = RealGenerator(SEED, call_fn=lambda s, u: "def solve(): pass")
    assert "def solve" in g2.generate("p", None, None).code

    # 4. 빈 응답 → base(seed) 폴백 (라운드 안 죽음)
    g3 = RealGenerator(SEED, call_fn=lambda s, u: "")
    assert g3.generate("p", None, None).code == SEED

    # 5. _extract_code 직접 — python 태그 유무
    assert _extract_code("```python\nX=1\n```") == "X=1"
    assert _extract_code("```\nY=2\n```") == "Y=2"
    assert _extract_code("no block here") == "no block here"

    print("generator.py self-check PASS — generate 흐름(추출·프롬프트·폴백). 실 API는 키 있을 때.")
