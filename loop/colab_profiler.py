"""colab-cli 프로파일러 (로컬측) — MailboxProfiler 대체.

설계: vault docs/10-colab-cli-migration-design.md. git-우편함(mailbox.py)의
REQ push→RES pull 왕복을 google-colab-cli(`colab exec`) 단일 blocking 호출로 치환.
git·폴링·watch·PAT 전부 소멸. 인터페이스(submit/submit_raw)는 MailboxProfiler와 동일 =
runner.MailboxGlue·run_gain_compare 무손상.

왕복 물리:
  로컬 cmd dict → JSON → `colab exec -s <sess> -f <wrapper>` (cmd를 stdin/env로)
  → Colab서 executor.execute_request(cmd) → RES dict → stdout 마지막 JSON 줄 → 로컬 파싱.

원격 전제: Colab 세션에 loop/ 코드가 있어야 함(executor import). setup_remote()가
`colab upload`로 1회 배포. 세션 수명은 호출자(run_gain_compare)가 colab new/stop 관리.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import subprocess

from glue import ProfileResult, GateResult
# MailboxResult 스키마 재사용(파싱 동일) — 상위가 .gate/.profile만 봄.
from mailbox import MailboxResult, MailboxTimeout


# 원격 wrapper 템플릿: cmd JSON을 코드에 임베드(colab exec는 stdin 코드 점유라
# 파이프로 cmd 못 줌 → 매 호출 wrapper에 박아 -f 실행). executor.execute_request 호출,
# 결과를 stdout `__RES__<json>` 줄로 출력. raw_script면 exec(watch._run_raw_script 계약).
_WRAPPER_TEMPLATE = r'''
import sys, json, io, contextlib, importlib
sys.path.insert(0, {remote_dir!r})
importlib.invalidate_caches()
cmd = json.loads({cmd_json!r})
if "raw_script" in cmd:
    ns = {{}}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(cmd["raw_script"], ns)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    res = json.loads(lines[-1]) if lines else {{"passed": False, "error": "no output"}}
    res.setdefault("id", cmd.get("id", "?"))
else:
    from executor import execute_request
    res = execute_request(cmd)
print("__RES__" + json.dumps(res, ensure_ascii=False))
'''


class ColabExecError(Exception):
    """colab exec 비영종료/파싱실패. MailboxTimeout 대역(라운드 스킵 신호)."""


class ColabExecProfiler:
    """code → colab exec → executor.execute_request → ProfileResult.

    MailboxProfiler 드롭인 대체. 같은 submit/submit_raw 시그니처.
    session = colab-cli 세션명(colab new -s <session>). remote_dir = Colab 내 loop/ 경로.
    run_fn = colab exec 실행 함수(테스트 주입용). 기본 = 실제 subprocess.
    """
    def __init__(
        self,
        session: str,
        remote_dir: str = "/content/loop",
        timeout_s: float = 600.0,
        run_fn: Callable[[str], str] | None = None,
    ):
        self.session = session
        self.remote_dir = remote_dir
        self.timeout_s = timeout_s
        self.run = run_fn or self._colab_exec

    def _colab_exec(self, wrapper_code: str) -> str:
        """cmd 임베드된 wrapper를 임시 .py로 써서 `colab exec -f` 실행 → stdout.

        colab exec는 -f FILE만 받고 stdin을 코드로 점유 → cmd는 wrapper에 임베드.
        --timeout 기본 30s는 측정에 부족 → self.timeout_s 명시.
        """
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".py")
        try:
            os.write(fd, wrapper_code.encode())
            os.close(fd)
            proc = subprocess.run(
                ["colab", "exec", "-s", self.session, "-f", path,
                 "--timeout", str(self.timeout_s)],
                capture_output=True, text=True, timeout=self.timeout_s + 60,
            )
        finally:
            os.unlink(path)
        if proc.returncode != 0:
            raise ColabExecError(
                f"colab exec rc={proc.returncode}: {proc.stderr[-500:]}")
        return proc.stdout

    def _dispatch(self, cmd: dict) -> dict:
        """cmd dict → wrapper 생성 → 원격 실행 → RES dict. 왕복 단일 지점."""
        wrapper = _WRAPPER_TEMPLATE.format(
            remote_dir=self.remote_dir,
            cmd_json=json.dumps(cmd, ensure_ascii=False),
        )
        out = self.run(wrapper)
        for ln in reversed(out.splitlines()):
            if ln.startswith("__RES__"):
                return json.loads(ln[len("__RES__"):])
        raise ColabExecError(f"no __RES__ line in output: {out[-500:]}")

    # ── MailboxProfiler 호환 인터페이스 ──

    def profile(self, code: str, problem: str) -> ProfileResult:
        return self.submit(code, problem).profile

    def submit(self, code: str, problem: str,
               profile_opts: dict | None = None) -> MailboxResult:
        cmd = {"id": "exec", "problem": problem, "code": code,
               "profile_opts": profile_opts or {"ncu": True}}
        return self._parse(self._dispatch(cmd))

    def submit_raw(self, raw_script: str, timeout_s: float | None = None) -> dict:
        cmd = {"id": "exec", "raw_script": raw_script}
        if timeout_s is not None:
            cmd["timeout_s"] = timeout_s
        return self._dispatch(cmd)

    @staticmethod
    def _parse(res: dict) -> MailboxResult:
        sig = dict(res.get("signal_dict") or {})
        lat = res.get("latency_us", sig.get("latency_us", 0.0))
        return MailboxResult(
            profile=ProfileResult(sig, lat),
            gate=GateResult(bool(res.get("passed", False)),
                            float(res.get("max_abs_err", 0.0))),
            error=res.get("error"),
        )


def setup_remote(session: str, local_loop_dir: str = ".",
                 remote_dir: str = "/content/loop",
                 run_fn: Callable[[list[str]], None] | None = None) -> None:
    """Colab 세션에 executor 의존 순수모듈 배포 (1회). executor import 가능하게.

    wrapper는 매 호출 cmd 임베드해 생성하므로 여기선 안 올림.
    run_fn = subprocess 실행자(테스트 주입). 기본 = 실제 colab upload.
    """
    def _up(local: str, remote: str):
        argv = ["colab", "upload", "-s", session, local, remote]
        if run_fn:
            run_fn(argv)
        else:
            subprocess.run(argv, check=True, capture_output=True, text=True)

    loop = Path(local_loop_dir)
    for name in ("signals.py", "glue.py", "executor.py"):
        _up(str(loop / name), f"{remote_dir}/{name}")


# ── self-check: colab-cli·GPU·git 0. run_fn 주입으로 왕복 검증. ──

if __name__ == "__main__":
    # 가짜 colab exec: 생성된 wrapper 코드를 실제 exec → 진짜 __RES__ 줄 나옴.
    # executor import는 없으니 raw_script 경로 + 임베드 cmd 파싱만 검증.
    # 표준 submit 경로는 wrapper가 executor.execute_request를 부르므로 stub 주입.
    def fake_run(wrapper_code: str) -> str:
        import io, contextlib
        # wrapper의 `from executor import execute_request`를 stub으로 가로챔.
        import types, sys as _sys
        stub = types.ModuleType("executor")
        stub.execute_request = lambda cmd: {
            "id": cmd["id"], "passed": True, "max_abs_err": 1e-6,
            "signal_dict": {"bw_pct": 0.48, "latency_us": 857.0},
            "latency_us": 857.0, "chip": "a100", "error": None}
        _sys.modules["executor"] = stub
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(wrapper_code, "<wrapper>", "exec"), {"__name__": "__main__"})
        return buf.getvalue()

    p = ColabExecProfiler("testsess", run_fn=fake_run)

    r = p.submit("# kernel", "llama_ffn")
    assert r.gate.passed and r.error is None, r
    assert r.profile.signal_dict["bw_pct"] == 0.48
    assert r.profile.latency_us == 857.0

    pr = p.profile("# k2", "llama_ffn")
    assert pr.latency_us == 857.0

    # raw_script 경로: wrapper가 raw exec → 마지막 JSON 줄 = RES (executor stub 불필요)
    raw = p.submit_raw("import json; print(json.dumps({'passed': True, 'chip': 'a100'}))",
                       timeout_s=5.0)
    assert raw["chip"] == "a100" and raw["passed"] is True, raw

    # __RES__ 줄 없으면 ColabExecError
    p_bad = ColabExecProfiler("s", run_fn=lambda w: "no res line here")
    try:
        p_bad.submit("# x", "p")
        raise AssertionError("expected ColabExecError")
    except ColabExecError:
        pass

    print("colab_profiler.py self-check PASS")
