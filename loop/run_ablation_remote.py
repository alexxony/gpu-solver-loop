"""ablation 원격 배치 드라이버 — ON/OFF 트랙 전체를 colab exec 1방으로 (A안).

문제(오늘 실측): run_gain_hypcond는 라운드마다 colab exec 왕복 → 12+7라운드에
벽시계 ~30분, 전송 스톨 노출 19회, GPU 유휴 99%. 룰 엔진은 결정론(LLM 0)이고
hypcond 콜백도 순수 dict 매핑이라 원격 실행에 제약 없음.

해결: loop 모듈 10종(전부 순수 파이썬)을 /content/loop에 배포하고, 문제 N개 ×
ON/OFF 트랙 전체를 **단일 colab exec**로 원격 실행. 라운드별 진행은 stdout,
최종 결과는 `__ABL_RESULT__<json>` + /content/abl_result.json(전송 유실 대비 이중화).
로컬은 결과 JSON을 TrackResult로 복원해 기존 _report로 판정 출력.

효과: 왕복 19회→1회, 세션 체류 ~30분→측정시간만, 전송 스톨 노출 1회.
LLM 대행 generate가 필요한 run_gain_round는 범위 밖(왕복 필수) — 이건 ablation 전용.

실행: python3 run_ablation_remote.py <problem>[,<problem>...] [max_rounds] --session=<s>
      [--skip-upload]  (모듈 이미 배포된 세션 재사용 시)
      [--selfcheck]    (GPU·colab 0 — 원격 드라이버 생성·컴파일 검증만)
variant 규약은 run_gain_hypcond와 동일: matmul=R_tf32on, 그 외=R_tf32+R_coalesced.
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from run_e2e import PROBLEMS
from run_gain_compare import TrackResult, _report

LOOP_DIR = Path(__file__).resolve().parent
REMOTE_DIR = "/content/loop"
# runner가 import하는 순수 모듈 전부 (executor 포함 — LocalProfiler가 직결).
MODULES = ("signals.py", "rules.py", "evolver.py", "ledger.py", "glue.py",
           "generator.py", "mailbox.py", "harness.py", "runner.py", "executor.py")

# ── 원격 드라이버 템플릿: 센티널 치환(.replace) — 코드 내 중괄호와 무충돌. ──
_REMOTE_TEMPLATE = r'''
import sys, json, importlib
sys.path.insert(0, __REMOTE_DIR__)
importlib.invalidate_caches()

CONFIG = json.loads(__CONFIG_JSON__)

import executor
from glue import ProfileResult, GateResult
from mailbox import MailboxResult
from runner import run_problem
from rules import seed_rules
from ledger import Ledger
from generator import CallbackGenerator
from signals import Context


class LocalProfiler:
    """submit → executor.execute_request 직결 (같은 프로세스, 왕복 0)."""
    def submit(self, code, problem, profile_opts=None):
        res = executor.execute_request({
            "id": "abl", "problem": problem, "code": code,
            "profile_opts": profile_opts or {"ncu": True}})
        sig = dict(res.get("signal_dict") or {})
        lat = res.get("latency_us", sig.get("latency_us", 0.0))
        return MailboxResult(
            profile=ProfileResult(sig, lat),
            gate=GateResult(bool(res.get("passed", False)),
                            float(res.get("max_abs_err", 0.0))),
            error=res.get("error"))

    def profile(self, code, problem):
        return self.submit(code, problem).profile


def _make_cb(seed_code, variant_map):
    def cb(problem, hyp, prev_code):
        label = hyp if isinstance(hyp, str) else None
        code = variant_map.get(label, seed_code) if label else seed_code
        which = label if (label in variant_map) else "seed(base)"
        print(f"  [generate] hyp={hyp!r} -> {which}", flush=True)
        return code
    return cb


def _run_track(problem, seed_code, variant_map, max_rounds, evolve_enabled,
               ledger_path, ctx):
    import os
    if os.path.exists(ledger_path):    # 이전 런 잔재가 곡선에 섞이는 것 방지
        os.unlink(ledger_path)
    rules = seed_rules()
    gen = CallbackGenerator(_make_cb(seed_code, variant_map))
    label = "evolve_ON" if evolve_enabled else "evolve_OFF"
    print(f"== {problem} / {label} ==", flush=True)
    res = run_problem(problem, seed_code, "/content/mb-unused", ledger_path,
                      sync_fn=lambda _p: None, max_rounds=max_rounds,
                      rules=rules, generator=gen, evolve_enabled=evolve_enabled,
                      metric_mode="latency", ctx=ctx, profiler=LocalProfiler())
    led = Ledger(ledger_path)
    recs = [r for r in led.records if r.problem == problem]
    return {
        "label": label,
        "metric_curve": led.metric_curve(problem),
        "fired_rules": [r.hypothesis_label for r in recs],
        "wasted_rounds": sum(1 for r in recs if r.passed and not r.improved),
        "retire_count": sum(1 for e in res.events if e.kind == "retire"),
        "stop_reason": res.stopped_reason,
        "rounds": res.rounds,
    }


def main():
    try:
        chip = executor._detect_chip_now()
    except Exception:
        chip = ""
    ctx = Context(chip=chip) if chip else None
    print(f"[remote] chip={chip!r} problems={list(CONFIG['problems'])} "
          f"max_rounds={CONFIG['max_rounds']}", flush=True)

    out = {"chip": chip, "results": {}}
    for name, spec in CONFIG["problems"].items():
        vmap = spec["variant_map"]
        out["results"][name] = {
            "off": _run_track(name, spec["seed"], vmap, CONFIG["max_rounds"],
                              False, f"/content/abl-{name}-off.jsonl", ctx),
            "on": _run_track(name, spec["seed"], vmap, CONFIG["max_rounds"],
                             True, f"/content/abl-{name}-on.jsonl", ctx),
        }
        with open("/content/abl_result.json", "w") as f:   # 중간에도 덮어씀
            json.dump(out, f)
    print("__ABL_RESULT__" + json.dumps(out), flush=True)


main()
'''


def _variant_map_for(problem: str) -> dict[str, str]:
    """run_gain_hypcond와 동일 규약 — 발화 룰 라벨 → 충실 variant 코드."""
    vdir = PROBLEMS / problem / "variants"
    if problem == "matmul":
        return {"fp32_no_tensorcore": (vdir / "R_tf32on.py").read_text()}
    return {
        "fp32_no_tensorcore": (vdir / "R_tf32.py").read_text(),
        "uncoalesced": (vdir / "R_coalesced.py").read_text(),
    }


def _build_remote_driver(problems: list[str], max_rounds: int) -> str:
    config = {"max_rounds": max_rounds, "problems": {}}
    for p in problems:
        seed = (PROBLEMS / p / "solve.py").read_text()
        config["problems"][p] = {"seed": seed, "variant_map": _variant_map_for(p)}
    return (_REMOTE_TEMPLATE
            .replace("__REMOTE_DIR__", repr(REMOTE_DIR))
            .replace("__CONFIG_JSON__",
                     repr(json.dumps(config, ensure_ascii=False))))


def _colab(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _retry(what: str, fn, attempts: int = 3):
    """세션 웜업 직후 HTTP ReadTimeout 등 전송 플레이크 흡수."""
    last = ""
    for i in range(1, attempts + 1):
        try:
            p = fn()
        except subprocess.TimeoutExpired:
            last = f"{what}: subprocess timeout (attempt {i}/{attempts})"
            print(f"  [deploy] {last} — 재시도", file=sys.stderr)
            continue
        if p.returncode == 0:
            return p
        last = f"{what}: rc={p.returncode} {p.stderr[-200:]} (attempt {i}/{attempts})"
        print(f"  [deploy] {last} — 재시도", file=sys.stderr)
    raise RuntimeError(f"재시도 소진 — {last}")


def _upload_modules(session: str) -> None:
    mk = f"import os; os.makedirs({REMOTE_DIR!r}, exist_ok=True); print('MKDIR OK')"
    _retry("mkdir", lambda: subprocess.run(
        ["colab", "exec", "-s", session, "--timeout", "120"], input=mk,
        capture_output=True, text=True, timeout=180))
    for name in MODULES:
        _retry(f"upload {name}", lambda n=name: _colab(
            ["colab", "upload", "-s", session,
             str(LOOP_DIR / n), f"{REMOTE_DIR}/{n}"], 120))
    print(f"[deploy] 모듈 {len(MODULES)}종 업로드 완료")


def _fetch_result_file(session: str) -> dict | None:
    """전송 유실 대비 — 원격 /content/abl_result.json 직접 회수."""
    code = "print(open('/content/abl_result.json').read())"
    p = subprocess.run(["colab", "exec", "-s", session], input=code,
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        return None
    for ln in reversed(p.stdout.splitlines()):
        if ln.strip().startswith("{"):
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                continue
    return None


def _to_track(d: dict) -> TrackResult:
    return TrackResult(d["label"], [tuple(x) for x in d["metric_curve"]],
                       d["fired_rules"], d["wasted_rounds"], d["retire_count"],
                       d["stop_reason"], d["rounds"])


def main() -> int:
    argv = sys.argv[1:]
    if "--selfcheck" in argv:
        src = _build_remote_driver(["kb_matmul_scalar"], 2)
        compile(src, "<remote_driver>", "exec")
        assert "__ABL_RESULT__" in src and "LocalProfiler" in src
        print("run_ablation_remote.py self-check PASS (원격 드라이버 생성·컴파일 OK)")
        return 0

    session = None
    skip_upload = "--skip-upload" in argv
    rest = []
    for a in argv:
        if a.startswith("--session="):
            session = a.split("=", 1)[1]
        elif a not in ("--skip-upload",):
            rest.append(a)
    if not session or not rest:
        print("usage: run_ablation_remote.py <p1>[,<p2>...] [max_rounds] "
              "--session=<s> [--skip-upload] [--selfcheck]", file=sys.stderr)
        return 2
    problems = rest[0].split(",")
    max_rounds = int(rest[1]) if len(rest) > 1 else 8

    if not skip_upload:
        _upload_modules(session)

    driver_src = _build_remote_driver(problems, max_rounds)
    total_rounds = len(problems) * 2 * max_rounds
    timeout = total_rounds * 150 + 300          # 라운드당 여유 150s + 부팅 여유

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(driver_src)
        path = f.name
    print(f"[launch] 문제 {problems} × ON/OFF × {max_rounds}R — "
          f"colab exec 1방 (timeout {timeout:.0f}s)")
    result = None
    try:
        p = _colab(["colab", "exec", "-s", session, "-f", path,
                    "--timeout", str(timeout)], timeout + 120)
        sys.stdout.write(p.stdout)
        for ln in reversed(p.stdout.splitlines()):
            if ln.startswith("__ABL_RESULT__"):
                result = json.loads(ln[len("__ABL_RESULT__"):])
                break
    except subprocess.TimeoutExpired:
        print("[warn] 전송 타임아웃 — 원격 결과 파일 회수 시도", file=sys.stderr)
    if result is None:
        result = _fetch_result_file(session)
    if result is None:
        print("ERR: 결과 회수 실패 (stdout·파일 둘 다)", file=sys.stderr)
        return 1

    print(f"\n[chip={result.get('chip')!r}]")
    for name, r in result["results"].items():
        print(f"\n################ {name} ################")
        _report(_to_track(r["on"]), _to_track(r["off"]), metric_mode="latency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
