"""Git-우편함 Runner (Colab측) — watch 루프. 우편함 반대편.

설계: docs/03-git-mailbox-runner.md §아키텍처. 로컬 MailboxProfiler가 push한
cmd/REQ-<id>.json을 소비 → 컴파일/gate/ncu 실행 → result/RES-<id>.json push.

Colab 셀 1개로 실행: `!python watch.py --loop` (git pull every poll_s).
GPU 실행부(execute_request)는 stub — Colab서 challenge reference_impl + ncu로 채움.
골격(폴링/멱등/스키마)은 GPU 없이 --selfcheck로 검증.

git 인증: PAT는 Colab Secrets → 환경변수. 이 파일엔 토큰 없음.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
import argparse
import json
import subprocess
import time
import traceback


# ── 메시지 I/O (mailbox.py와 동일 스키마, 반대 방향) ──

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(mb: Path, *args: str) -> None:
    """git 명령 1회. 실패 시 예외(워치 루프가 잡아 계속)."""
    subprocess.run(["git", "-C", str(mb), *args], check=True,
                   capture_output=True, text=True)


def git_sync_push(mb: Path) -> None:
    """pull(로컬 REQ 받기) + push(RES 보내기). 충돌 없음: cmd/ vs result/ 분리."""
    _git(mb, "pull", "-q", "--no-edit")
    # 변경 있을 때만 commit (없으면 git이 비영 종료)
    st = subprocess.run(["git", "-C", str(mb), "status", "--porcelain"],
                        capture_output=True, text=True).stdout.strip()
    if st:
        _git(mb, "add", "-A")
        _git(mb, "commit", "-q", "-m", "watch: results")
        _git(mb, "push", "-q")


# ── GPU 실행부 ──
# 실구현은 executor.execute_request (torch/ncu 필요 → Colab). 여기선 lazy import:
# torch 있으면 실구현을 기본 execute로, 없으면(self-check 등) stub.

def execute_request(cmd: dict) -> dict:
    """REQ → RES. executor.execute_request로 위임. torch 부재 시 stub.

    실구현 흐름(executor): cmd['code'](전체 solve.py) → _reference 교차검증 gate →
    ncu 프로파일(or Event fallback) → signal_dict. 반환 = RES 스키마.

    cmd['raw_script'] 있으면 = 임의 파이썬을 subprocess 실행(executor 우회).
    새 측정 로직을 REQ에 데이터로 실어 보냄 → executor/watch 코드 안 고쳐
    Colab 재시작 영원 불요 (sys.modules 캐시 회피). PROGRESS 운용교훈의 근본해결.
    """
    if "raw_script" in cmd:
        return _run_raw_script(cmd)
    try:
        from . import executor
    except ImportError:
        import executor                  # Colab !python 직접 실행 대비
    return executor.execute_request(cmd)


def _run_raw_script(cmd: dict) -> dict:
    """cmd['raw_script'](파이썬 소스)를 subprocess 실행 → stdout 마지막 JSON 줄을 RES로.

    스크립트는 마지막 줄에 RES 스키마 dict를 한 줄 JSON으로 print해야 함.
    timeout·비영종료·JSON파싱실패는 _error_result로 격리(라운드 스킵, watch 생존).
    """
    rid = cmd.get("id", "raw")
    try:
        proc = subprocess.run(
            ["python3", "-c", cmd["raw_script"]],
            capture_output=True, text=True,
            timeout=cmd.get("timeout_s", 600),
        )
    except subprocess.TimeoutExpired:
        return _error_result(rid, "raw_script timeout")
    if proc.returncode != 0:
        return _error_result(rid, f"raw_script exit {proc.returncode}\n{proc.stderr[-2000:]}")
    # stdout 마지막 비공백 줄 = RES JSON (스크립트 진단 print 허용)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return _error_result(rid, f"raw_script no output\n{proc.stderr[-2000:]}")
    try:
        res = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return _error_result(rid, f"raw_script bad JSON: {e}\nlast={lines[-1][:500]}")
    res.setdefault("id", rid)
    return res


def _error_result(rid: str, err: str) -> dict:
    """실행 실패 시 RES — 라운드 스킵용. gate 실패와 구분(passed=False+error)."""
    return {"id": rid, "passed": False, "max_abs_err": 0.0,
            "signal_dict": {}, "latency_us": 0.0, "error": err}


# ── watch 루프 ──

def process_pending(
    mailbox_dir: str | Path,
    execute: Callable[[dict], dict] = execute_request,
) -> int:
    """미처리 REQ 전부 처리 → RES 생성. done/ 마커로 멱등. 처리 건수 반환.

    한 REQ가 죽어도(execute 예외) _error_result로 RES 만들고 계속 — 한 라운드
    실패가 watch 전체를 멈추지 않게.
    """
    mb = Path(mailbox_dir)
    cmd_dir, res_dir, done_dir = mb / "cmd", mb / "result", mb / "done"
    n = 0
    reqs = sorted(cmd_dir.glob("REQ-*.json")) if cmd_dir.exists() else []
    for req in reqs:
        rid = req.stem[len("REQ-"):]
        if (done_dir / rid).exists():
            continue
        try:
            result = execute(_read_json(req))
        except Exception:
            result = _error_result(rid, traceback.format_exc(limit=3))
        result.setdefault("id", rid)
        _write_json(res_dir / f"RES-{rid}.json", result)
        done_dir.mkdir(parents=True, exist_ok=True)
        (done_dir / rid).write_text("")
        n += 1
    return n


def watch_loop(
    mailbox_dir: str | Path,
    poll_s: float = 5.0,
    sync_fn: Callable[[Path], None] = git_sync_push,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_iters: int | None = None,     # None = 무한 (Colab). 테스트는 유한.
    execute: Callable[[dict], dict] = execute_request,
) -> None:
    """git pull → process_pending → push 반복. Colab 셀에서 무한 실행.

    execute 기본값 = executor 위임(실 GPU). self-check는 stub 주입해 GPU 0 유지.
    """
    mb = Path(mailbox_dir)
    i = 0
    while max_iters is None or i < max_iters:
        try:
            sync_fn(mb)                    # pull REQ (+ 이전 RES push)
            done = process_pending(mb, execute=execute)
            if done:
                sync_fn(mb)                # RES 즉시 push
        except Exception:
            traceback.print_exc()          # 일시 git 실패 등 — 죽지 않고 계속
        sleep_fn(poll_s)
        i += 1


# ── self-check: GPU·git 0. 폴링/멱등/실패격리/스키마. ──

def _selfcheck() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        mb = Path(d)
        # 로컬측이 REQ 2개 push한 상황 모킹
        _write_json(mb / "cmd" / "REQ-a.json",
                    {"id": "a", "problem": "p", "code": "# k"})
        _write_json(mb / "cmd" / "REQ-b.json",
                    {"id": "b", "problem": "p", "code": "# k2"})

        # 가짜 GPU 실행: bw 따라 정상 RES
        def fake_exec(cmd):
            assert "code" in cmd
            return {"passed": True, "max_abs_err": 1e-6,
                    "signal_dict": {"bw_pct": 0.48, "weight_pct": 1.0,
                                    "tensorcore_active": True, "latency_us": 857.0},
                    "latency_us": 857.0, "error": None}

        n = process_pending(d, execute=fake_exec)
        assert n == 2, n
        ra = _read_json(mb / "result" / "RES-a.json")
        assert ra["passed"] and ra["signal_dict"]["bw_pct"] == 0.48
        assert ra["id"] == "a"

        # 멱등: 재실행 시 done 마커로 0건
        assert process_pending(d, execute=fake_exec) == 0

        # 실패 격리: execute가 예외 던져도 RES(error) 생성, 죽지 않음
        _write_json(mb / "cmd" / "REQ-c.json", {"id": "c", "code": "# bad"})
        def boom(cmd): raise RuntimeError("ncu crashed")
        assert process_pending(d, execute=boom) == 1
        rc = _read_json(mb / "result" / "RES-c.json")
        assert rc["passed"] is False and "ncu crashed" in rc["error"]

    # watch_loop: 유한 반복 + sync/sleep/execute 주입 (git·GPU·실시간 0)
    with tempfile.TemporaryDirectory() as d2:
        mb2 = Path(d2)
        _write_json(mb2 / "cmd" / "REQ-x.json", {"id": "x", "code": "# k"})
        calls = {"sync": 0, "sleep": 0}
        def syncf(_m): calls["sync"] += 1
        def sleepf(_s): calls["sleep"] += 1
        # 기본 execute는 executor(torch/ncu) 위임 → self-check는 stub 주입해 GPU 0.
        def stub_exec(cmd):
            raise NotImplementedError("self-check stub")
        watch_loop(d2, poll_s=0.0, sync_fn=syncf, sleep_fn=sleepf,
                   max_iters=2, execute=stub_exec)
        assert calls["sleep"] == 2
        assert calls["sync"] >= 2               # pull 2회 + 처리 후 push
        # stub 예외 → _error_result 경유. passed=False지만 RES 존재(실패 격리).
        assert (mb2 / "result" / "RES-x.json").exists()
        rx = _read_json(mb2 / "result" / "RES-x.json")
        assert rx["passed"] is False and "NotImplementedError" in rx["error"]

    # raw_script 분기: executor 우회, subprocess stdout 마지막 JSON 줄 = RES
    ok = execute_request({"id": "r1", "raw_script":
        "print('진단 로그 무시됨'); import json; "
        "print(json.dumps({'passed': True, 'latency_us': 42.0, 'signal_dict': {'occupancy': 0.7}}))"})
    assert ok["passed"] and ok["latency_us"] == 42.0 and ok["id"] == "r1", ok
    assert ok["signal_dict"]["occupancy"] == 0.7
    # 비영 종료 → _error_result 격리
    bad = execute_request({"id": "r2", "raw_script": "import sys; sys.exit(3)"})
    assert bad["passed"] is False and "exit 3" in bad["error"], bad
    # JSON 없는 출력 → bad JSON 격리
    nj = execute_request({"id": "r3", "raw_script": "print('not json')"})
    assert nj["passed"] is False and "bad JSON" in nj["error"], nj

    print("watch.py self-check PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Git-우편함 Colab측 watch")
    ap.add_argument("--mailbox", default=".", help="우편함 repo 경로")
    ap.add_argument("--poll", type=float, default=5.0, help="폴링 간격(초)")
    ap.add_argument("--once", action="store_true", help="1회만 처리 후 종료")
    ap.add_argument("--loop", action="store_true", help="무한 watch (Colab)")
    ap.add_argument("--selfcheck", action="store_true", help="GPU 없이 골격 검증")
    a = ap.parse_args()

    if a.selfcheck:
        _selfcheck()
    elif a.once:
        n = process_pending(a.mailbox)
        print(f"processed {n} request(s)")
    elif a.loop:
        print(f"watching {a.mailbox} every {a.poll}s … (Ctrl-C to stop)")
        watch_loop(a.mailbox, poll_s=a.poll)
    else:
        ap.print_help()
