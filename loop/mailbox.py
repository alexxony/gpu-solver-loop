"""Git-우편함 Runner (로컬측) — glue.Profiler 구현.

설계: docs/03-git-mailbox-runner.md. 터널/SSH/URL 제거, cmd/result JSON을
git repo로 비동기 교환. 여기 = 로컬측 클라이언트(MailboxProfiler).
Colab측 watch.py는 별도(GPU 필요, Colab서 작성).

GPU·git·네트워크 없이 테스트: sync_fn(git pull/push)을 주입.
가짜 우편함 = 로컬 폴더 1개 + no-op sync. self-check가 전체 왕복 검증.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import time
import uuid

from glue import ProfileResult, GateResult


# ── 메시지 I/O (cmd/ ↔ result/ 디렉토리) ──

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@dataclass
class MailboxResult:
    """RES JSON 파싱 결과. gate+profile 한 왕복에 합침."""
    profile: ProfileResult
    gate: GateResult
    error: str | None = None


class MailboxTimeout(Exception):
    """timeout_s 내 RES 안 옴 (Colab 죽음 등). 라운드 스킵/재큐 신호."""


class MailboxProfiler:
    """code → cmd/REQ push → result/RES pull → ProfileResult.

    sync_fn(mailbox_dir) = git pull+push 1회 (실제 운용). 테스트는 no-op.
    Colab watch가 같은 폴더 반대편(cmd 소비 → result 생성)을 처리.
    """
    def __init__(
        self,
        mailbox_dir: str | Path,
        sync_fn: Callable[[Path], None] = lambda _d: None,
        poll_s: float = 5.0,
        timeout_s: float = 600.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], float] = time.monotonic,
        id_fn: Callable[[], str] = lambda: uuid.uuid4().hex,
    ):
        self.mb = Path(mailbox_dir)
        self.sync = sync_fn
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.sleep = sleep_fn
        self.now = now_fn
        self.id_fn = id_fn

    # glue.Profiler Protocol
    def profile(self, code: str, problem: str) -> ProfileResult:
        return self.submit(code, problem).profile

    def submit(self, code: str, problem: str,
               profile_opts: dict | None = None) -> MailboxResult:
        rid = self.id_fn()
        _write_json(self.mb / "cmd" / f"REQ-{rid}.json", {
            "id": rid,
            "problem": problem,
            "code": code,
            "profile_opts": profile_opts or {"ncu": True},
        })
        self.sync(self.mb)                       # push REQ
        res = self._await_result(rid)            # poll until RES
        return self._parse(res)

    def _await_result(self, rid: str) -> dict:
        res_path = self.mb / "result" / f"RES-{rid}.json"
        deadline = self.now() + self.timeout_s
        while True:
            self.sync(self.mb)                   # pull (Colab의 RES push 받음)
            if res_path.exists():
                return _read_json(res_path)
            if self.now() >= deadline:
                raise MailboxTimeout(f"no RES-{rid} within {self.timeout_s}s")
            self.sleep(self.poll_s)

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


# ── 가짜 Colab watch (테스트 전용) — cmd 소비 → result 생성 ──

def fake_colab_respond(mailbox_dir: str | Path,
                       responder: Callable[[dict], dict]) -> int:
    """우편함의 모든 미처리 REQ를 responder로 처리해 RES 생성.

    responder(cmd_dict) → result_dict. 실제 Colab watch.py의 로컬 모킹.
    반환 = 처리한 건수. done/ 마커로 재처리 방지.
    """
    mb = Path(mailbox_dir)
    cmd_dir, res_dir, done_dir = mb / "cmd", mb / "result", mb / "done"
    n = 0
    for req in sorted(cmd_dir.glob("REQ-*.json")) if cmd_dir.exists() else []:
        rid = req.stem[len("REQ-"):]
        if (done_dir / rid).exists():
            continue
        result = responder(_read_json(req))
        result.setdefault("id", rid)
        _write_json(res_dir / f"RES-{rid}.json", result)
        (done_dir).mkdir(parents=True, exist_ok=True)
        (done_dir / rid).write_text("")
        n += 1
    return n


# ── self-check: GPU·git·네트워크 0. 전체 왕복 + 타임아웃. ──

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        # 가짜 Colab: bw_pct 따라 latency 깎고 통과 판정.
        def responder(cmd: dict) -> dict:
            assert "code" in cmd and "problem" in cmd
            return {
                "passed": True,
                "max_abs_err": 1e-6,
                "signal_dict": {"bw_pct": 0.48, "weight_pct": 0.03,
                                "tensorcore_active": True, "latency_us": 857.0},
                "latency_us": 857.0,
                "error": None,
            }

        calls = {"sync": 0}

        def sync_fn(_mb):
            calls["sync"] += 1
            # 실제 git pull 시점에 Colab이 응답한다고 가정 → 첫 pull 전에 처리.
            fake_colab_respond(d, responder)

        mp = MailboxProfiler(d, sync_fn=sync_fn, poll_s=0.0,
                             id_fn=lambda: "fixedid")

        r = mp.submit("# kernel src", "llama_ffn")
        assert r.gate.passed and r.error is None
        assert r.profile.signal_dict["bw_pct"] == 0.48
        assert r.profile.latency_us == 857.0
        assert calls["sync"] >= 2                 # push + 최소 1 pull

        # Profiler Protocol 경로
        pr = mp.profile("# k2", "llama_ffn")
        assert pr.latency_us == 857.0

        # 멱등: done 마커로 재처리 안 함 (REQ 2건이지만 RES도 2건만)
        import glob, os
        assert len(glob.glob(os.path.join(d, "result", "RES-*.json"))) == 1  # fixedid 덮어씀
        assert len(glob.glob(os.path.join(d, "done", "*"))) == 1

    # 타임아웃: Colab 무응답 → MailboxTimeout
    with tempfile.TemporaryDirectory() as d2:
        ticks = iter([0.0, 0.0, 100.0, 200.0])     # now_fn: deadline 초과 유도
        mp2 = MailboxProfiler(d2, sync_fn=lambda _m: None, poll_s=0.0,
                              timeout_s=50.0, sleep_fn=lambda _s: None,
                              now_fn=lambda: next(ticks))
        try:
            mp2.submit("# x", "p")
            raise AssertionError("expected MailboxTimeout")
        except MailboxTimeout:
            pass

    print("mailbox.py self-check PASS")
