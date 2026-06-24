"""A 인프라 e2e 드라이버 — 진짜 우편함 왕복 1라운드 (로컬측).

설계: docs/03-git-mailbox-runner.md §재개절차 4. runner.run_problem에 진짜
git sync_fn(pull+push)을 주입 → REQ push → Colab watch가 GPU 작업 → RES pull.
self-check(__main__ fake)와 달리 여기는 실제 git/GPU 왕복.

전제: Colab서 watch.watch_loop 떠 있어야 (셀5). mailbox/loop 둘 다 clone됨.
실행: python run_e2e.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

# loop 디렉토리에서 직접 실행 가정 (Colab !python 대비 runner와 동일 패턴)
from runner import run_problem
from ledger import Ledger

MAILBOX = Path.home() / "workspace" / "gpu-mailbox"
SOLVE = Path(__file__).resolve().parents[1] / "solve.py"
LEDGER = MAILBOX.parent / "e2e-ledger.jsonl"
PROBLEM = "solve_llama"
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
    g("pull", "-q", "--rebase", "origin", BRANCH)
    g("push", "-q", "origin", BRANCH)


def main() -> int:
    if not (MAILBOX / ".git").exists():
        print(f"ERR: mailbox clone 없음: {MAILBOX}", file=sys.stderr)
        return 2
    if not SOLVE.exists():
        print(f"ERR: solve.py 없음: {SOLVE}", file=sys.stderr)
        return 2

    seed = SOLVE.read_text()
    print(f"e2e 시작 — problem={PROBLEM} mailbox={MAILBOX} branch={BRANCH}")
    print(f"  seed=solve.py ({len(seed)} chars), ledger={LEDGER}")
    print("  Colab watch 떠 있어야 함. REQ push → RES 대기...")

    # 고정 코드 → metric 정체 → 빠른 수렴. 1라운드면 배관 실증 충분.
    res = run_problem(PROBLEM, seed, MAILBOX, LEDGER,
                      sync_fn=git_sync, max_rounds=1,
                      poll_s=5.0, timeout_s=900.0)

    led = Ledger(str(LEDGER))
    recs = [r for r in led.records if r.problem == PROBLEM]
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
