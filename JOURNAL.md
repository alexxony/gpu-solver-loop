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
- 2026-07-24T06:20:00+09:00 — `index.html` 전면 재작성(portfolio 3장 시리즈 1호, CSS·구조 확정 기준). 영문 단일화(기존 한/영 혼용 제거). 구조: ①상단(정의+숫자3+architecture.svg) ②ablation 토글(evolution ON/OFF, kb_matmul_scalar 실측값 하드코딩: ON 55.5us/retire@R4/uncoalesced 발화, OFF 319.1us/영구 오탐) ③gain 증거(matmul 6.4×) ④endpoint 증거(5.75×, 3회 재현) ⑤retire 증거(T4 8R) ⑥한계 정식 섹션(5개 항목, 축소 없음) ⑦하단(repo 링크 + 이웃 compiler-thermal·hbm-build GitHub 절대 URL — 두 repo 모두 Pages 미설정 확인되어 repo URL로 링크, Pages URL 날조 안 함). 게이트: `grep kimsh/LG-PC//home/` 0건, 한글 grep 0건, JS `node --check` 통과, 외부 CDN/localStorage 0건, SVG 4개 참조·실재 확인, 태그 밸런스 확인.
- 2026-07-24T06:30:39+09:00 — `docs/METHOD.md` 신규 작성(03-DEV-METHOD 감사 결론에 따른 공개용 방법론 문서, method-doc 에이전트 위임). 좁힌 5규율(ON/OFF 통제 비교·원장 즉시 기록·음성 결과 보고·사전등록+반증 기록·재구성 실패 판별) 각 1문단 + 세 repo 중 증거 있는 실경로 병기(hbm-build/compiler-thermal 인용, gpu-solver-loop 자체는 사전등록·검증자 격리 증거 없음을 "In this repo" 절에 명시). C 목록 6건 축약("What this process caught") + 미해결 이월 1건("What it did not catch": G4 B-series 0.7616) 포함. 검증자 격리 문단은 구조적 이유(컨텍스트 오염·앵커링·목적함수 충돌) 서술, gpu-solver-loop에는 해당 증거 없음을 "later projects in this stack"으로 정확히 한정. README §3 끝에 링크 추가. 게이트: 한글 grep 0건, PII grep 0건(`docs/METHOD.md`, README.md 수정부).
- 2026-07-24T07:05:00+09:00 — README.md H1 아래 배지 행 1줄 추가(awesome-readme 대조 리뷰 채택분, gsl-badge 에이전트 위임). License(MIT — LICENSE 파일 실확인) + Python(3 — 63개 .py 파일, `python3` CLI 전반 사용 실확인) 2개만 채택. Tests 배지는 미채택: `find`로 `test_*.py`/`*_test.py` 부재 확인(`test.ipynb` 1건만 존재, pytest 스위트 없음), README 어디에도 테스트 수 기재 없어 발명 방지 원칙에 따라 생략. Status/simulated 성격 배지도 미채택(숫자 아닌 주장성 배지 배제 방침). 게이트: 한글 grep 0건, PII 0건(순수 ASCII shields.io 배지 2개).
