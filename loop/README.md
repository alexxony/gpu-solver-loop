# GPU-Solver — 자동 최적화 루프 골격

Agentic GPU kernel optimizer. 설계: vault `GPU-Solver/` (`2026-06-22-agentic-gpu-optimizer-design.md`).

## 컴포넌트 (★ = 내 코드, 나머지 = 교체 가능 글루)

| 파일 | 역할 | spec |
|---|---|---|
| `signals.py` | ★Trace Parser — ncu CSV → 정규화 Signal | §3.1 |
| `rules.py` | ★Hypothesis Engine — 시드룰 6개 + 매칭 | §3.2 |
| `evolver.py` | **★★Rule Evolver — 유일 차별점 (룰 진화 메타루프)** | §3.3 |
| `ledger.py` | Run Ledger — 라운드 JSONL 적재 (곡선 원천) | §2-7 |
| `harness.py` | 오케스트레이터 — gen→gate→profile→hyp→ledger→evolve | §2 |
| `glue.py` | Generator/Gate/Profiler stub + FakeGlue | §2-1·2·3 |
| `selfcheck.py` | 통합 + 차별점 E2E | — |

## 검증

```bash
python3 selfcheck.py   # 전부 GPU 없이 도는 self-check
```

차별점 증명: 틀린 정적 임계값이 측정 피드백으로 신뢰도 강등(0.5→0.0)→폐기→
다음 후보 전환. CUDAMaster류 정적 룰은 불변 — 격차 실증.

## 상태

- ✅ 순수 로직 (★4·5·6·7) 완결, 로컬 self-check PASS
- ⏭️ 글루 (1·2·3) = stub. Colab에서 RealGenerator(LLM)/RealGate(challenge.py
  reference_impl 비교)/RealProfiler(ncu)로 교체 필요 (GPU+API 키).
