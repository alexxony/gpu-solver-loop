"""성능 gain 드라이버 — 가설-조건부 콜백 (run_gain_compare의 큐 한계 극복).

문제: run_gain_compare는 variant를 라운드 순서 큐로 고정 → ON/OFF가 같은 코드 봄
  → latency 동일 → 성능 gain(latency 갈림) 구조적으로 안 나옴 (그 파일 line 17-22 경고).

해결: 콜백이 '발화 룰 라벨'을 보고 코드 선택 (큐 아님).
  - hyp="fp32_no_tensorcore" → TF32 variant (groupnorm엔 matmul 없음 → null, latency 불변)
  - hyp="uncoalesced"        → 합착 variant (welford 1패스 + 작은 BLOCK → 진짜 빠름)
  - 그 외/R0                 → seed (base)

이러면 진화 차이가 코드 차이로 이어짐 (공정: 양쪽 같은 콜백 로직):
  ON : fp32 헛발화→TF32(null)→demote 누적→retire→uncoalesced 발화→합착→빨라짐.
  OFF: fp32 영원 발화→TF32(null) 영원→영원 같은 latency.
  → ON best latency < OFF best latency = 성능 gain (진화→더 빠른 커널).

⚠️ 공정성: ON/OFF 모두 동일 콜백·동일 variant 풀. 차이는 오직 evolver(retire)가
  발화 룰을 바꾸는 데서만 나옴. 그게 차별점 분리.

실행: python run_gain_hypcond.py <problem> [max_rounds]
  전제: Colab watch 살아있음 + mailbox/problems clone. variants/<problem>/R_*.py 존재.
"""
from __future__ import annotations
import sys
from pathlib import Path

from runner import run_problem
from ledger import Ledger
from rules import seed_rules
from generator import CallbackGenerator
from run_e2e import git_sync, MAILBOX, PROBLEMS
from run_gain_compare import TrackResult, _report


def _make_hypcond_callback(seed_code: str, variant_map: dict[str, str]):
    """발화 룰 라벨 → 코드 매핑 콜백. hyp 없으면(R0) seed.

    harness._last_hyp가 직전 라운드 hypothesis_label을 hyp.prompt로 넘김
    (prompt 아니라 라벨 문자열). 그걸 variant_map 키로 매칭.
    """
    def cb(problem, hyp, prev_code):
        # hyp = 직전 발화 룰 라벨 (문자열) 또는 None (R0).
        label = hyp if isinstance(hyp, str) else None
        code = variant_map.get(label, seed_code) if label else seed_code
        which = label if (label in variant_map) else "seed(base)"
        print(f"  [generate] 직전가설={hyp!r} → 코드={which}")
        return code
    return cb


def _run_track(label, problem, seed_code, variant_map, max_rounds,
               evolve_enabled, ledger_path, profiler=None):
    if Path(ledger_path).exists():
        Path(ledger_path).unlink()
    rules = seed_rules()
    gen = CallbackGenerator(_make_hypcond_callback(seed_code, variant_map))
    print(f"\n=== 트랙 {label} (evolve={'ON' if evolve_enabled else 'OFF'}) ===")
    res = run_problem(problem, seed_code, MAILBOX, ledger_path,
                      sync_fn=git_sync, max_rounds=max_rounds, poll_s=5.0,
                      timeout_s=900.0, rules=rules, generator=gen,
                      evolve_enabled=evolve_enabled, metric_mode="latency",
                      profiler=profiler)
    led = Ledger(str(ledger_path))
    recs = [r for r in led.records if r.problem == problem]
    curve = led.metric_curve(problem)
    fired = [r.hypothesis_label for r in recs]
    wasted = sum(1 for r in recs if r.passed and not r.improved)
    retires = sum(1 for e in res.events if e.kind == "retire")
    return TrackResult(label, curve, fired, wasted, retires,
                       res.stopped_reason, res.rounds)


def main() -> int:
    profiler, use_colab_cli, argv = make_colab_profiler(sys.argv[1:])
    problem = argv[0] if len(argv) > 0 else "groupnorm"
    max_rounds = int(argv[1]) if len(argv) > 1 else 8

    seed_path = PROBLEMS / problem / "solve.py"
    vdir = PROBLEMS / problem / "variants"
    if not use_colab_cli and not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음 {MAILBOX}", file=sys.stderr); return 2
    seed_code = seed_path.read_text()

    if problem == "matmul":
        # matmul = fp32_no_tensorcore가 맞는 룰: 발화 → TF32 켜면 진짜 6.4× gain.
        # 루프가 측정→가설→재작성→더 빠른 커널 도달함을 ON curve 하강으로 입증.
        tf32on = vdir / "R_tf32on.py"
        if not tf32on.exists():
            print(f"ERR: 없음 {tf32on}", file=sys.stderr); return 2
        variant_map = {"fp32_no_tensorcore": tf32on.read_text()}
    else:
        tf32 = vdir / "R_tf32.py"
        coal = vdir / "R_coalesced.py"
        for p in (tf32, coal):
            if not p.exists():
                print(f"ERR: 없음 {p}", file=sys.stderr); return 2
        # 발화 룰 라벨 → 충실 variant. cherry-pick 아님: 각 룰 가설에 맞는 코드.
        variant_map = {
            "fp32_no_tensorcore": tf32.read_text(),   # matmul 없음 → null
            "uncoalesced": coal.read_text(),           # load_eff=0 고침 → gain
        }

    print(f"성능 gain (가설-조건부) — {problem}, max_rounds={max_rounds}")
    print(f"  variant_map: {list(variant_map)} (seed=base)")
    print(f"  ON=fp32 retire→uncoalesced→합착→빨라짐 / OFF=fp32 영원 TF32 null\n")

    base = MAILBOX.parent / f"gain-hypcond-{problem}"
    off = _run_track("evolve_OFF", problem, seed_code, variant_map,
                     max_rounds, False, f"{base}-off.jsonl", profiler=profiler)
    on = _run_track("evolve_ON", problem, seed_code, variant_map,
                    max_rounds, True, f"{base}-on.jsonl", profiler=profiler)
    _report(on, off, metric_mode="latency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
