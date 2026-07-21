# JOURNAL — gpu-solver-loop 원장

> 2층 기록 관례(2026-07-18 승인): 이 파일 = 이벤트 즉시 append + 원자 커밋(원장).
> 마감 요약은 vault `GPU-Solver/PROGRESS.md`. 세션 재개 정본 = git log + 이 파일 tail.

## 2026-07-21

- 2026-07-21T19:40:00+09:00 — repo public 전환: `gh api PATCH repos/alexxony/gpu-solver-loop private=false` → `"private": false` 확인. HEAD `8307a06` = 2026-07-12 민감정보 전이력 스캔(CLEAN) 시점과 동일(이후 커밋 0) → 스캔 유효 상태로 공개.
- 2026-07-21T19:52:00+09:00 — GitHub Pages 활성화: `gh api POST repos/.../pages` JSON body(`build_type=legacy, source=master /`) → status `built` → https://alexxony.github.io/gpu-solver-loop/ HTTP 200 라이브. ⚠️ `-f "source[branch]=..."` 폼 방식은 422 — JSON body(`--input -`) 필수.
- 2026-07-21T20:01:00+09:00 — JOURNAL.md 원장 시작(2층 기록 관례 GPU-Solver 첫 적용). 코드 변경 없음(이번 세션 = repo 공개 절차만).
