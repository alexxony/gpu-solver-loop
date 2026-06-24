"""GPU 실행부 (Colab측) — execute_request 실구현. watch.py가 주입.

설계: docs/03-git-mailbox-runner.md §컴포넌트매핑. cmd["code"](전체 solve.py)
→ correctness gate(_reference 교차검증) → ncu 프로파일(or Event fallback)
→ signals.parse_ncu_rows → RES dict.

watch.py와 분리 이유: 우편함 로직(폴링/멱등/git) ≠ GPU 작업(gate/ncu).
watch.py는 GPU 0으로 self-check 가능해야 하므로 torch import를 여기 격리.

ncu fallback: ncu 권한/부재 시 torch.cuda.Event로 latency만 측정(s1-165 Phase 0).
signal_dict는 latency_us + weight_pct=1.0만 — 메트릭 축소, 배관은 굴러감.
"""
from __future__ import annotations
from pathlib import Path
import csv
import importlib.util
import io
import subprocess
import sys
import tempfile

# signals는 같은 패키지. ncu CSV → Signal 정규화.
try:
    from . import signals
except ImportError:                      # 스크립트 직접 실행(Colab !python) 대비
    import signals

# gate가 도는 seq_len — solve.py _check와 동일 (단/짧음/중간/타깃).
GATE_SEQ_LENS = (1, 4, 128, 2048)
GATE_ATOL = 1e-3
GATE_RTOL = 1e-3

# ncu가 뽑을 메트릭 = signals.NCU_METRIC_MAP의 키 전부.
NCU_METRICS = ",".join(signals.NCU_METRIC_MAP.keys())


def _load_solve_module(code: str):
    """cmd['code'](전체 solve.py 텍스트) → 임포트된 모듈. 임시파일 경유.

    solve.py는 import 시 torch.set_float32_matmul_precision 등 전역 설정 실행 →
    파일로 써서 정상 import. 반환 모듈서 solve/_reference/_make_case/D 꺼냄.
    """
    tmp = tempfile.NamedTemporaryFile("w", suffix="_round.py", delete=False)
    tmp.write(code)
    tmp.close()
    spec = importlib.util.spec_from_file_location("_solve_round", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)         # solve.py __main__ 가드 → import 시 실행 안 됨
    return mod, tmp.name


def _run_gate(mod) -> tuple[bool, float]:
    """solve(mod.solve)를 mod._reference와 교차검증. (passed, max_abs_err).

    solve.py에 이미 있는 _make_case/_reference 재사용 — 다른 코드 경로라 교차검증
    유효. GATE_SEQ_LENS 전부 PASS해야 passed=True.
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    worst_err = 0.0
    ok = True
    for T in GATE_SEQ_LENS:
        x, w, cos, sin = mod._make_case(T, dev)
        out = torch.empty(T, mod.D, device=dev)
        mod.solve(x, out, w, cos, sin, T)
        ref = mod._reference(x, w, cos, sin, T)
        err = (out - ref).abs().max().item()
        worst_err = max(worst_err, err)
        ok &= torch.allclose(out, ref, atol=GATE_ATOL, rtol=GATE_RTOL)
    return ok, worst_err


def _profile_ncu(code_path: str) -> dict | None:
    """ncu로 code_path --profile 실행 → signal_dict. ncu 부재/실패 시 None.

    ncu --csv 출력을 signals.parse_ncu_rows로 정규화. 단일 모듈 가정 →
    weight_pct=1.0 (전체가 이 커널). 여러 커널이면 후속 harness가 합산.
    """
    cmd = [
        "ncu", "--metrics", NCU_METRICS, "--csv",
        "--target-processes", "all",
        sys.executable, code_path, "--profile",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    rows = _parse_ncu_csv(r.stdout)
    if not rows:
        return None
    sig = signals.parse_ncu_rows(rows)
    d = {k: getattr(sig, k) for k in signals.Signal.__annotations__}
    d["weight_pct"] = 1.0                # 단일 모듈 = 전체 비중
    return d


def _parse_ncu_csv(stdout: str) -> list[dict]:
    """ncu --csv stdout → [{'Metric Name','Metric Value'}, ...].

    ncu csv는 메트릭당 1행, 'Metric Name'/'Metric Value' 컬럼 포함. 헤더 앞에
    배너 줄이 섞일 수 있어 'Metric Name' 헤더 줄부터 파싱.
    """
    lines = stdout.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Metric Name" in ln), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return [row for row in reader if row.get("Metric Name")]


def _profile_event(mod) -> dict:
    """ncu fallback — torch.cuda.Event로 latency만. signal_dict 축소.

    A100서 ncu 권한 없을 때(s1-165 Phase 0 fallback). 메트릭은 못 뽑지만
    latency_us + weight_pct=1.0으로 배관은 굴러감. bw_pct 등은 0(미측정).
    """
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = 2048
    x, w, cos, sin = mod._make_case(T, dev)
    out = torch.empty(T, mod.D, device=dev)
    if dev == "cpu":                     # GPU 없으면 latency 의미 없음
        return {"latency_us": 0.0, "weight_pct": 1.0}
    for _ in range(10):                  # warm-up
        mod.solve(x, out, w, cos, sin, T)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    iters = 50
    for _ in range(iters):
        mod.solve(x, out, w, cos, sin, T)
    e.record()
    torch.cuda.synchronize()
    ms_per = s.elapsed_time(e) / iters
    return {"latency_us": ms_per * 1000.0, "weight_pct": 1.0}


def execute_request(cmd: dict) -> dict:
    """REQ → RES. cmd['code'](전체 solve.py) → gate → 프로파일 → RES dict.

    1. code → 모듈 로드 (전역 설정 실행).
    2. correctness gate: _reference 교차검증 (GATE_SEQ_LENS 전부 PASS).
    3. gate FAIL이면 프로파일 스킵 (RES passed=False).
    4. gate PASS면 ncu 프로파일, 실패 시 Event fallback.
    5. RES = {id, passed, max_abs_err, signal_dict, latency_us, error}.

    예외는 watch.process_pending이 잡아 _error_result로 → infra 실패와 구분.
    """
    rid = cmd.get("id", "?")
    code = cmd["code"]
    mod, path = _load_solve_module(code)

    passed, max_err = _run_gate(mod)
    if not passed:
        return {"id": rid, "passed": False, "max_abs_err": max_err,
                "signal_dict": {}, "latency_us": 0.0,
                "error": f"correctness gate FAIL (max_err={max_err:.2e})"}

    sig = _profile_ncu(path)
    if sig is None:                      # ncu 부재/실패 → Event fallback
        sig = _profile_event(mod)
    latency = sig.get("latency_us", 0.0)
    return {"id": rid, "passed": True, "max_abs_err": max_err,
            "signal_dict": sig, "latency_us": latency, "error": None}


if __name__ == "__main__":
    # self-check: GPU/ncu/torch 없이 도는 부분만 — CSV 파싱 + ncu-None 분기.
    # gate(_run_gate)·Event 타이밍은 torch 필요 → Colab 실라운드서 검증.

    # 1. ncu CSV 파싱: 배너 줄 + 헤더 혼재 → Metric 행만 추출
    csv_out = (
        '==PROF== banner line\n'
        '"ID","Metric Name","Metric Value"\n'
        '"0","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed","83.0"\n'
        '"0","gpu__time_duration.sum","857000"\n'
    )
    rows = _parse_ncu_csv(csv_out)
    assert len(rows) == 2, rows
    s = signals.parse_ncu_rows(rows)
    assert abs(s.bw_pct - 0.83) < 1e-9, s.bw_pct
    assert s.latency_us == 857000.0, s.latency_us

    # 2. 빈/헤더없는 출력 → 빈 리스트
    assert _parse_ncu_csv("garbage\nno header here") == []

    # 3. ncu 부재/파일없음 → None (Event fallback 트리거)
    assert _profile_ncu("/nonexistent_xyz.py") is None

    # 4. NCU_METRICS = 매핑 키 전부 (ncu 호출 인자 무결성)
    assert "gpu__time_duration.sum" in NCU_METRICS
    assert NCU_METRICS.count(",") == len(signals.NCU_METRIC_MAP) - 1

    print("executor.py self-check PASS (CSV파싱·ncu-None분기. gate/Event는 Colab torch로)")
