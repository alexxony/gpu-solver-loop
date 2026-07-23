# JOURNAL — gpu-solver-loop 원장

> 2층 기록 관례(2026-07-18 승인): 이 파일 = 이벤트 즉시 append + 원자 커밋(원장).
> 마감 요약은 vault `GPU-Solver/PROGRESS.md`. 세션 재개 정본 = git log + 이 파일 tail.

## 2026-07-21

- 2026-07-21T19:40:00+09:00 — repo public 전환: `gh api PATCH repos/alexxony/gpu-solver-loop private=false` → `"private": false` 확인. HEAD `8307a06` = 2026-07-12 민감정보 전이력 스캔(CLEAN) 시점과 동일(이후 커밋 0) → 스캔 유효 상태로 공개.
- 2026-07-21T19:52:00+09:00 — GitHub Pages 활성화: `gh api POST repos/.../pages` JSON body(`build_type=legacy, source=master /`) → status `built` → https://alexxony.github.io/gpu-solver-loop/ HTTP 200 라이브. ⚠️ `-f "source[branch]=..."` 폼 방식은 422 — JSON body(`--input -`) 필수.
- 2026-07-21T20:01:00+09:00 — JOURNAL.md 원장 시작(2층 기록 관례 GPU-Solver 첫 적용). 코드 변경 없음(이번 세션 = repo 공개 절차만).
- 2026-07-21T20:30:00+09:00 — KernelBench(ScalingIntelligence) 단독 조사·등재: 판정 = 선행 경쟁자 아니라 문제 공급원(순수 벤치마크, 룰 진화·병목 판정 0, "not agentic scaffolds" 명시). 이미 kb_* 4문제로 사용 중이던 repo. 250 확장 소비 경로 = clone/submodule 또는 HF dataset(pip 아님). vault 02-prior-art-survey §KernelBench 자체 신설(d1cc544).
- 2026-07-21T20:35:00+09:00 — 세션 마감: 코드 변경 0, 미추적 3건(.claude/, problems/2d_convolution/starter.cu, scratchpad/) 처리 보류. 다음 = (b) KernelBench 250 확장(추천) / (c) CudaForge 비교 / (d) 두 축 재설계.
- 2026-07-22T14:43:45+09:00 — LICENSE(MIT) 추가: compiler-thermal 공개 전환 작업과 함께 두 repo 일괄 결정(사용자 승인). 라이선스 공백(all rights reserved) 해소.
- 2026-07-24T05:40:00+09:00 — README 뼈대 재구성(세 저장소 공통 6섹션 시범 적용 1호): 순서만 재배열, 내용 삭제 없음. ①정의+숫자 ②왜 필요한가(이웃 프로젝트 compiler-thermal·hbm-build 본문 언급) ③어떻게(architecture.svg 최상단 배치) ④증거(헤드라인별 출처+재현 명령) ⑤한계(정식 섹션 승격, "reproduced 3 independent runs"=동일 절차 재실행이며 격리 검증자 아님을 명시) ⑥상태(페이즈 라벨 P0~P7은 유지하되 본문 제목에서는 제거). `charts/architecture.svg` 신규 추가(mmdc 렌더링, exit 0, 오렌지 피드백 엣지 `#d79b00` 3건 확인).
