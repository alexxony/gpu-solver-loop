"""gain layer 드라이버 — 진화 ON vs OFF 다라운드 비교 (gain 입증/반증).

설계: GPU-Solver/docs/04-multiproblem-round-design.md §gain layer.
run_gain_round(2라운드 1점) → N라운드 + ON/OFF 비교로 확장한 본 드라이버.

핵심: "루프 돌면 개선되냐"(mechanism 닫힘)를 넘어 "진화가 정적보다 이득이냐"(gain).
  - mechanism = 이미 입증 (B 대행 GPU 닫힘, sigmoid 2R).
  - gain      = 같은 variant 시퀀스를 진화 ON·OFF 두 트랙에 주입 → 두 곡선 비교.
                ON이 헛라운드↓ / 수렴↑ / best↑ 면 진화 이득. 같으면 반증.

왜 사전 큐잉(variants 파일)인가:
  CallbackGenerator 콜백은 라운드별 맥락(가설·prev_code) 보고 변형코드를 줘야 함.
  비대화형 1프로세스로 N라운드 자동 못 돎(매 라운드 대행 개입 필요) →
  라운드별 변형 solve.py를 미리 variants/<problem>/R0.py..RN.py 로 준비.
  같은 큐를 ON/OFF 양쪽에 주입 = generate를 결정론화 → "진화만의 차이" 분리 (공정).

⚠️ 공정성 경계: ON/OFF가 같은 코드 시퀀스를 보므로 측정 신호도 동일.
  → 두 트랙 차이는 오직 evolver(룰 진화)에서만 나옴. 그게 차별점 분리의 핵심.
  단, 현 harness에서 evolve는 "발화 룰 선택"에 영향을 주지(폐기→다른 룰) 코드
  생성엔 영향 없음 → ON/OFF 곡선 자체(metric)는 같을 수 있음. 차이는 "어떤 룰이
  발화했나 / 헛 가설 몇 번 / retire 일어났나"로 나타남. gain 지표를 metric 곡선이
  아니라 [헛라운드 수, retire 수, 발화룰 다양성]으로 본다 (정직한 정의).

실행:
  실 GPU: python run_gain_compare.py <problem> <variants_dir> [max_rounds]
          전제: Colab watch 살아있음 + mailbox clone.
  self-check(GPU·git 0): python run_gain_compare.py --selfcheck
"""
from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass

from runner import run_problem
from ledger import Ledger
from rules import seed_rules
from generator import CallbackGenerator


@dataclass
class TrackResult:
    """한 트랙(ON 또는 OFF) 1회 실행 결과 — gain 지표 원천."""
    label: str                   # "evolve_ON" | "evolve_OFF"
    metric_curve: list           # [(round_idx, metric)]
    fired_rules: list            # 라운드별 발화 룰 라벨
    wasted_rounds: int           # improved=False 라운드 수 (헛 가설)
    retire_count: int            # 폐기 이벤트 수 (ON만 >0 가능)
    stop_reason: str
    rounds: int


def _make_queued_callback(variant_codes: list[str], seed_code: str):
    """라운드별 변형코드를 순서대로 반환하는 콜백 (사전 큐잉).

    R0 = seed 원본(base), R1.. = variant_codes[0..]. 큐 소진 후엔 마지막 코드 유지
    (수렴/종료 라운드서 generate가 또 불려도 안 죽게).
    """
    calls = {"n": 0}
    def cb(problem, hyp, prev_code):
        i = calls["n"]; calls["n"] += 1
        print(f"  [generate R{i}] 가설={hyp!r}")
        if i == 0:
            return seed_code
        idx = i - 1
        return variant_codes[idx] if idx < len(variant_codes) else variant_codes[-1]
    return cb


def _run_track(label: str, problem: str, seed_code: str, variant_codes: list[str],
               mailbox_dir, ledger_path, sync_fn, max_rounds: int,
               evolve_enabled: bool, poll_s: float, timeout_s: float) -> TrackResult:
    """한 트랙 실행 — 같은 variant 큐, evolve_enabled만 다름."""
    if Path(ledger_path).exists():
        Path(ledger_path).unlink()
    rules = seed_rules()                  # 트랙마다 새 룰판 (오염 방지)
    cb = _make_queued_callback(variant_codes, seed_code)
    gen = CallbackGenerator(cb)

    print(f"\n=== 트랙 {label} (evolve={'ON' if evolve_enabled else 'OFF'}) ===")
    res = run_problem(problem, seed_code, mailbox_dir, ledger_path,
                      sync_fn=sync_fn, max_rounds=max_rounds, poll_s=poll_s,
                      timeout_s=timeout_s, rules=rules, generator=gen,
                      evolve_enabled=evolve_enabled)

    led = Ledger(str(ledger_path))
    recs = [r for r in led.records if r.problem == problem]
    curve = led.metric_curve(problem)
    fired = [r.hypothesis_label for r in recs]
    wasted = sum(1 for r in recs if r.passed and not r.improved)
    retires = sum(1 for e in res.events if e.kind == "retire")
    return TrackResult(label, curve, fired, wasted, retires,
                       res.stopped_reason, res.rounds)


def _report(on: TrackResult, off: TrackResult) -> None:
    print("\n" + "=" * 56)
    print("gain 비교 — 진화 ON vs OFF")
    print("=" * 56)
    for t in (off, on):
        ms = [round(m, 4) for _, m in t.metric_curve]
        print(f"\n[{t.label}] rounds={t.rounds} stop={t.stop_reason}")
        print(f"  metric 곡선 : {ms}")
        print(f"  발화 룰     : {t.fired_rules}")
        print(f"  헛라운드    : {t.wasted_rounds}")
        print(f"  retire 수   : {t.retire_count}")

    print("\n--- 판정 ---")
    gain_signals = []
    if on.retire_count > off.retire_count:
        gain_signals.append(f"retire ON={on.retire_count} > OFF={off.retire_count} (틀린 룰 폐기)")
    if on.wasted_rounds < off.wasted_rounds:
        gain_signals.append(f"헛라운드 ON={on.wasted_rounds} < OFF={off.wasted_rounds} (덜 헤맴)")
    if on.fired_rules != off.fired_rules:
        gain_signals.append("발화 룰 시퀀스 다름 (진화가 룰 선택 바꿈)")

    if gain_signals:
        print("  ✅ gain 신호:")
        for g in gain_signals:
            print(f"     - {g}")
    else:
        print("  ❌ gain 신호 없음 — ON/OFF 동일. 이 variant 큐론 진화 이득 미관찰.")
        print("     (variant가 틀린 룰을 반복 발화시키게 설계돼야 retire가 갈림)")
    print("\n⚠️ 경계: metric 곡선은 같을 수 있음(같은 코드 큐). gain은 룰 진화 지표")
    print("   [retire/헛라운드/발화 다양성]으로 본다. 곡선 우상향=mechanism, 진화차이=gain.")


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--selfcheck":
        return _selfcheck()

    if len(sys.argv) < 3:
        print("usage: python run_gain_compare.py <problem> <variants_dir> [max_rounds]",
              file=sys.stderr)
        print("       python run_gain_compare.py --selfcheck", file=sys.stderr)
        return 2

    from run_e2e import git_sync, MAILBOX, PROBLEMS

    problem = sys.argv[1]
    variants_dir = Path(sys.argv[2])
    max_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    seed_path = PROBLEMS / problem / "solve.py"
    if not seed_path.exists():
        print(f"ERR: seed 없음 {seed_path}", file=sys.stderr); return 2
    if not variants_dir.is_dir():
        print(f"ERR: variants 디렉터리 없음 {variants_dir}", file=sys.stderr); return 2
    if not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음 {MAILBOX}", file=sys.stderr); return 2

    # variants/<problem>/R1.py, R2.py ... 순서 로드 (R0=seed라 R1부터).
    # 'R<숫자>.py'만 (R0_seed_ref 등 비-순수숫자 stem 제외).
    import re
    def _rnum(p):
        m = re.fullmatch(r"R(\d+)", p.stem)
        return int(m.group(1)) if m else None
    variant_files = sorted([p for p in variants_dir.glob("R*.py") if _rnum(p) is not None],
                           key=_rnum)
    if not variant_files:
        print(f"ERR: variants 없음 (R*.py) in {variants_dir}", file=sys.stderr); return 2
    variant_codes = [f.read_text() for f in variant_files]
    seed_code = seed_path.read_text()

    print(f"gain 비교 — {problem}, variants={[f.name for f in variant_files]} "
          f"max_rounds={max_rounds}")
    print(f"  같은 variant 큐를 ON/OFF 두 트랙에 주입 (공정 비교).\n")

    led_base = MAILBOX.parent / "gain-compare"
    off = _run_track("evolve_OFF", problem, seed_code, variant_codes,
                     MAILBOX, f"{led_base}-off.jsonl", git_sync, max_rounds,
                     evolve_enabled=False, poll_s=5.0, timeout_s=900.0)
    on = _run_track("evolve_ON", problem, seed_code, variant_codes,
                    MAILBOX, f"{led_base}-on.jsonl", git_sync, max_rounds,
                    evolve_enabled=True, poll_s=5.0, timeout_s=900.0)
    _report(on, off)
    return 0


def _selfcheck() -> int:
    """GPU·git·LLM 0 — fake 우편함으로 ON/OFF 비교 로직 검증.

    설계: variant가 '틀린 룰을 반복 발화시키는' 신호를 내게 만들어, ON 트랙선
    retire가 일어나고 OFF 트랙선 안 일어남을 확인 = gain 신호 갈림 검증.
    """
    import tempfile, os
    from mailbox import fake_colab_respond

    # responder: 비-stop 룰(uncoalesced)을 반복 발화시키되 improved 안 되는 신호.
    #   STOP_LABELS = {memory_saturated, below_weight_gate, tensorcore_saturated} 회피:
    #     weight_pct>=0.05(게이트통과) · compute_tput=0(fp32룰 회피) ·
    #     tensorcore_active=False · bw_pct<=0.5(fusable/saturated 회피) ·
    #     load_eff<0.7 → uncoalesced(priority4, 비-stop) 발화.
    #   occupancy=0 고정 → metric(=bw_pct) 정체 → improved=False 반복 → demote→retire.
    def responder(cmd):
        return {"passed": True, "max_abs_err": 1e-6,
                "signal_dict": {"occupancy": 0.0, "bw_pct": 0.30,
                                "compute_tput": 0.0, "weight_pct": 1.0,
                                "load_eff": 0.5, "tensorcore_active": False,
                                "latency_us": 80.0},
                "latency_us": 80.0, "error": None}

    with tempfile.TemporaryDirectory() as d:
        def sync_fn(_mb):
            fake_colab_respond(d, responder)

        seed = "# seed solver\ndef solve(): pass\n"
        # 6개 더미 variant (코드 내용은 측정에 무관 — responder 고정). 큐 작동만 검증.
        variants = [f"# variant R{i}\ndef solve(): pass  # v{i}\n" for i in range(1, 7)]

        off = _run_track("evolve_OFF", "selfprob", seed, variants, d,
                         os.path.join(d, "off.jsonl"), sync_fn, max_rounds=8,
                         evolve_enabled=False, poll_s=0.0, timeout_s=10.0)
        on = _run_track("evolve_ON", "selfprob", seed, variants, d,
                        os.path.join(d, "on.jsonl"), sync_fn, max_rounds=8,
                        evolve_enabled=True, poll_s=0.0, timeout_s=10.0)

        _report(on, off)

        # 검증: 두 트랙 모두 라운드 돎 (큐 주입 작동)
        assert off.rounds >= 1 and on.rounds >= 1, "두 트랙 다 돌아야"
        assert len(off.metric_curve) >= 1, "OFF 곡선 기록돼야"
        # OFF는 절대 retire 안 함 (evolve_enabled=False)
        assert off.retire_count == 0, f"OFF retire=0이어야, got {off.retire_count}"
        # ON은 반복 실패 신호로 retire 일어나야 (차별점) — fail>=3 & n>=4 조건 충족하도록
        # max_rounds=8이면 발화룰이 충분히 demote됨.
        assert on.retire_count >= 1, f"ON retire>=1이어야 (gain 신호), got {on.retire_count}"
        # gain 신호: ON retire > OFF retire = 진화 이득 분리됨
        assert on.retire_count > off.retire_count, "ON이 OFF보다 retire 많아야 (gain 갈림)"
        print("\nrun_gain_compare.py self-check PASS — ON/OFF 비교 로직 + gain 신호 갈림 검증")
    return 0


if __name__ == "__main__":
    sys.exit(main())
