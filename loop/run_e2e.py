"""A 인프라 e2e 드라이버 — 진짜 우편함 왕복 1라운드 (로컬측).

설계: docs/03-git-mailbox-runner.md §재개절차 4. runner.run_problem에 진짜
git sync_fn(pull+push)을 주입 → REQ push → Colab watch가 GPU 작업 → RES pull.
self-check(__main__ fake)와 달리 여기는 실제 git/GPU 왕복.

전제: Colab서 watch.watch_loop 떠 있어야 (셀5). mailbox/loop 둘 다 clone됨.
실행: python run_e2e.py
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

# loop 디렉토리에서 직접 실행 가정 (Colab !python 대비 runner와 동일 패턴)
from runner import run_problem
from ledger import Ledger

MAILBOX = Path.home() / "workspace" / "gpu-mailbox"
PROBLEMS = Path(__file__).resolve().parents[1] / "problems"
LEDGER = MAILBOX.parent / "e2e-ledger.jsonl"
BRANCH = "main"   # mailbox repo 기본 브랜치 (loop은 master, mailbox은 main — 주의)


def git_sync(mb: Path) -> None:
    """우편함 1왕복: 로컬 변경(REQ) commit+push → remote(RES) pull.

    submit이 REQ-*.json 쓴 뒤 이 함수 호출. 순서:
      add → commit(변경 있으면) → pull --rebase(RES 받기) → push(REQ 보내기).
    pull을 push 앞에 둬 divergent 방지. capture로 노이즈 억제하되 실패는 raise.
    """
    def g(*args, check=True):
        r = subprocess.run(["git", "-C", str(mb), *args],
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {args[0]} 실패 (rc={r.returncode}):\n{r.stderr}")
        return r

    g("add", "-A")
    # 스테이지에 변경 있을 때만 commit (없으면 commit이 rc=1 → check=False)
    staged = g("diff", "--cached", "--quiet", check=False)
    if staged.returncode != 0:
        g("commit", "-q", "-m", "local: REQ/cleanup")
    # pull→push 사이 watch가 RES push하면 레이스(non-fast-forward) → 재시도.
    # 매 시도: pull --rebase(remote 변경 흡수) 후 push. 최대 5회 (s2-231).
    for attempt in range(5):
        g("pull", "-q", "--rebase", "origin", BRANCH)
        r = g("push", "-q", "origin", BRANCH, check=False)
        if r.returncode == 0:
            return
    raise RuntimeError(f"git push 5회 재시도 실패:\n{r.stderr}")


def make_colab_profiler(argv):
    """드라이버 공통 — argv서 --colab-cli [--session=<name>] 파싱 → ColabExecProfiler|None.

    None이면 기존 git-우편함(MailboxProfiler). 반환 (profiler, use_colab_cli, clean_argv).
    clean_argv = --colab-cli/--session 제거된 argv (드라이버 위치인자 파싱 보존).
    design 10. mailbox clone 체크는 use_colab_cli일 때 스킵해야.
    """
    use = "--colab-cli" in argv
    session = next((a.split("=", 1)[1] for a in argv if a.startswith("--session=")),
                   "gpucanary")
    clean = [a for a in argv
             if a != "--colab-cli" and not a.startswith("--session=")]
    prof = None
    if use:
        from colab_profiler import ColabExecProfiler
        prof = ColabExecProfiler(session, remote_dir="/content/loop", timeout_s=900.0)
        print(f"[colab-cli] session={session} — git-우편함 우회, colab exec 직결")
    return prof, use, clean


def main() -> int:
    profiler, use_colab_cli, argv = make_colab_profiler(sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", nargs="?", default="llama",
                    help="problems/ 하위 폴더명 (llama/sigmoid/groupnorm)")
    args = ap.parse_args(argv)
    problem = args.problem
    solve_path = PROBLEMS / problem / "solve.py"

    # colab-cli 직결이면 mailbox clone 불필요 (design 10).
    if not use_colab_cli and not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음: {MAILBOX}", file=sys.stderr)
        return 2
    if not solve_path.exists():
        print(f"ERR: solve.py 없음: {solve_path}", file=sys.stderr)
        return 2

    seed = solve_path.read_text()
    if use_colab_cli:
        print(f"e2e 시작 — problem={problem} (colab exec 직결)")
        print(f"  seed={solve_path} ({len(seed)} chars), ledger={LEDGER}")
    else:
        print(f"e2e 시작 — problem={problem} mailbox={MAILBOX} branch={BRANCH}")
        print(f"  seed={solve_path} ({len(seed)} chars), ledger={LEDGER}")
        print("  Colab watch 떠 있어야 함. REQ push → RES 대기...")

    # 고정 코드 → metric 정체 → 빠른 수렴. 1라운드면 배관 실증 충분.
    res = run_problem(problem, seed, MAILBOX, LEDGER,
                      sync_fn=git_sync, max_rounds=1,
                      poll_s=5.0, timeout_s=900.0, profiler=profiler)

    led = Ledger(str(LEDGER))
    recs = [r for r in led.records if r.problem == problem]
    print(f"\n=== e2e 결과 ===")
    print(f"rounds={res.rounds} stop={res.stopped_reason} events={len(res.events)}")
    if not recs:
        print("FAIL: ledger에 라운드 기록 없음")
        return 1
    r0 = recs[-1]
    print(f"passed={r0.passed}")
    print(f"signal_dict={r0.signal}")
    # 검증점: RES-real01(bw_pct 0.524 등)과 일치하면 배관 정확
    print("\n검증: bw_pct·tensorcore_active 등이 수동 PoC 챔피언과 일치하는지 확인.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
