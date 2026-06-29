"""Hypothesis Engine (★내 코드) — 시드룰 매칭 → 병목 라벨 + 변형 프롬프트.

설계: design spec §3.2. 각 룰에 "왜 이 신호→이 병목" 근거 1줄 필수.
근거 없으면 룩업테이블, 있으면 perf 이해.

시드룰 = spec 표 그대로 (CUDAMaster 30% 임계값 + NVIDIA roofline 차용 근거).
이 표는 Rule Evolver(evolver.py)가 신뢰도를 갱신하는 '진화의 출발점'.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from signals import Signal, Context, DEFAULT_CTX

SLOW_LATENCY_US = 50.0  # "느림" 임계 (PoC FFN mul 76μs 기준 근방). evolver가 조정 가능.
WEIGHT_GATE = 0.05      # ≥5% 비중 게이트 (PoC R3': 커널 -39%였으나 비중 2.5%<5% → 전체 1%).
                        # evolver가 조정 가능한 임계. trajectory가 검증한 핵심 룰.


@dataclass
class Rule:
    label: str                       # 병목 라벨
    cond: Callable[[Signal], bool]   # 발화 조건 (신호만 — 환경 가드는 chip_cap이 분리)
    prompt: str                      # LLM에 줄 변형 가설 프롬프트
    rationale: str                   # 왜 이 신호→이 병목 (근거 1줄, 필수)
    priority: int                    # 낮을수록 먼저 (지배 병목 우선)
    # 환경 가드 (design [[07-chip-lang-context-design]]): 이 룰이 요구하는 칩 capability.
    # ""=칩 무관(항상 가능). "tf32"=TF32 있는 칩서만 발화 (T4/V100서 헛가설 차단).
    # 신호 cond와 분리한 이유: lambda 11개 안 건드리고 가드를 선언적 1필드로 (회귀 0).
    chip_cap: str = ""
    # evolver가 갱신하는 진화 상태
    success: int = 0
    fail: int = 0
    retired: bool = False
    seed: bool = True                # 시드룰인가, evolver가 추가한 후보인가

    @property
    def confidence(self) -> float:
        """성공률. 시도 없으면 시드 신뢰 0.5 (탐색 여지)."""
        n = self.success + self.fail
        return 0.5 if n == 0 else self.success / n


def seed_rules() -> list[Rule]:
    """시드룰 = PoC R3~R7 trajectory가 측정으로 검증한 룰 + 일반 CUDA 룰.

    trajectory 룰(priority 0~2)이 일반 룰(3~)보다 우선 — 이 워크로드선 dtype/
    텐서코어/비중 게이트가 지배적이었음. 임계값(WEIGHT_GATE 등)은 evolver가 조정 가능.
    """
    return [
        # ── priority 0: 게이트 (먼저 거른다 — 헛 타깃 차단) ──
        Rule(
            label="below_weight_gate",
            cond=lambda t: 0 < t.weight_pct < WEIGHT_GATE,
            prompt="STOP: 비중 < 5%. 커널 크게 줄여도 전체 이득 묻힘 (R3' -39%→전체1%).",
            rationale="전체 이득 ≤ 커널 비중 (Amdahl). 비중<게이트면 최적화 무의미",
            priority=0,
        ),
        # ── priority 1: 텐서코어 (이 워크로드 최대 이득 R5, 최대 함정 bf16) ──
        Rule(
            label="fp32_no_tensorcore",
            cond=lambda t: t.weight_pct >= WEIGHT_GATE and t.compute_tput > 0
            and not t.tensorcore_active,
            prompt="matmul 정밀도 TF32로: torch.set_float32_matmul_precision('high'). "
                   "fp32 sgemm은 텐서코어 미사용 (R5: matmul 52.7%→20.3%, 전체 1.71×).",
            rationale="fp32 GEMM = CUDA코어, 텐서코어 놀고있음 → TF32로 태우면 대폭↓",
            priority=1,
            chip_cap="tf32",   # TF32 있는 칩(A100/H100)서만. T4/V100엔 TF32 없음 → 헛가설 차단.
        ),
        Rule(
            label="tensorcore_saturated",
            cond=lambda t: t.tensorcore_active and t.latency_us > SLOW_LATENCY_US,
            prompt="STOP: 이미 텐서코어 경로. 저정밀(bf16) 추가 가속 거의 없음, "
                   "정확성만 악화 (R6 반증: TF32 0.853 vs bf16 0.850, 동률).",
            rationale="A100 TF32 throughput ≈ bf16 (같은 TC 파이프) → 저정밀 무효",
            priority=1,
            chip_cap="tf32",   # "TF32≈bf16" 근거가 TF32 칩 전제. 비-TF32 칩선 이 STOP 부적용.
        ),
        # ── priority 2: 융합 (게이트 통과한 메모리바운드만, 약 이득 R3') ──
        Rule(
            label="memory_bound_fusable",
            cond=lambda t: t.weight_pct >= WEIGHT_GATE and t.bw_pct > 0.5
            and not t.tensorcore_active,
            prompt="elementwise 연쇄를 1커널 Triton 융합 (silu*up류). matmul은 cuBLAS 유지. "
                   "주의: 융합해도 launch 오버헤드로 승격 안 될 수 있음 — 실측 비교 필수.",
            rationale="메모리바운드 = HBM 왕복 천장 → 융합으로 중간텐서 왕복 제거",
            priority=2,
        ),
        # ── priority 2: launch 오버헤드 (nsys 타임라인 신호 — ncu per-kernel론 안 보임) ──
        Rule(
            label="launch_overhead",
            cond=lambda t: t.launch_gap_pct > 0.3 and 0 < t.weight_pct < 0.5,
            prompt="GPU idle 큼(커널 간 gap) + 단일 지배커널 없음 → 작은 커널 다수. "
                   "CUDA graph 캡처 또는 elementwise 융합으로 launch 횟수↓.",
            rationale="nsys: GPU idle>30% & 지배커널 없음 = launch/CPU 바운드 (ncu per-kernel론 안 보임)",
            priority=2,
        ),
        # ── priority 3+: 일반 CUDA 룰 (다른 워크로드/커널 대비, 보존) ──
        Rule(
            label="memory_saturated",
            cond=lambda t: t.bw_pct > 0.8,
            prompt="STOP: 대역폭 포화, 손댈 것 없음",
            rationale="elementwise는 BW가 천장 (NVIDIA roofline: AI<ridge → mem bound)",
            priority=3,
        ),
        # torch profiler 신호 — 연산자 의미 귀속 (ncu 커널명 수작업 분류 대체).
        # attention 지배인데 다른 커널 손대면 헛수고 (llama: attention 54%>matmul 23% 교훈).
        Rule(
            label="attention_dominant",
            cond=lambda t: t.op_weight > 0.4
            and ("attention" in t.op_name.lower() or "sdpa" in t.op_name.lower()
                 or "fmha" in t.op_name.lower()),
            prompt="attention이 지배 연산자(>40%). flash/SDPA 백엔드 확인 — fp32 폴백이면 "
                   "bf16 flash로. matmul/TF32 튜닝은 비병목이라 무효.",
            rationale="torch op 귀속: attention self-time>40% = 여기가 병목, 타 커널 손대도 헛수고",
            priority=3,
        ),
        Rule(
            label="uncoalesced",
            cond=lambda t: t.load_eff < 0.7,
            prompt="인덱싱 재배열 + shared 타일링으로 합착 접근 복구",
            rationale="비합착 접근 = 한 워프가 여러 캐시라인 → 대역폭 낭비",
            priority=4,
        ),
        Rule(
            label="reg_pressure",
            cond=lambda t: t.occ < 0.5 and t.reg > 64,
            prompt="스레드 수↓ 또는 __launch_bounds__로 레지스터 압박 완화",
            rationale="점유율 제한 = 레지스터 (SM당 레지스터 유한 → 동시 워프↓)",
            priority=5,
        ),
    ]


@dataclass
class Hypothesis:
    label: str
    prompt: str
    rationale: str
    rule_idx: int           # 어느 룰이 발화했나 (ledger/evolver 추적용)
    is_stop: bool = False   # STOP 판정인가 (포화군)


def match(sig: Signal, rules: list[Rule], ctx: Context | None = None) -> Hypothesis | None:
    """가장 지배적 병목 1개 선택 (spec §3.2: 우선순위 → 신뢰도).

    탐색-활용: 같은 priority면 confidence 높은 룰 우선 (evolver 신뢰도 반영).
    retired 룰은 건너뜀.

    ctx (design [[07-chip-lang-context-design]]): 환경 가드. None=DEFAULT_CTX(칩 미지=모두 통과,
    종전 A100 동작 보존). 룰의 chip_cap이 비고 그 칩이 능력 없으면 발화 차단 (T4서 TF32 룰 등).
    cond(신호)와 chip_cap(환경)을 AND — 신호 맞아도 칩이 못 하면 헛가설이라 끔.
    """
    ctx = ctx or DEFAULT_CTX
    live = [(i, r) for i, r in enumerate(rules)
            if not r.retired and ctx.cap(r.chip_cap) and r.cond(sig)]
    if not live:
        return None
    # priority 오름차순, 동률이면 confidence 내림차순
    i, r = min(live, key=lambda ir: (ir[1].priority, -ir[1].confidence))
    return Hypothesis(
        label=r.label, prompt=r.prompt, rationale=r.rationale,
        rule_idx=i, is_stop=r.label in STOP_LABELS,
    )


# STOP 라벨 = 더 손댈 것 없는 포화/무효 (정직한 종료). trajectory 반증서 도출.
STOP_LABELS = {"memory_saturated", "below_weight_gate", "tensorcore_saturated"}


if __name__ == "__main__":
    from signals import from_dict
    rules = seed_rules()

    # self-check 1: 비중<5% → below_weight_gate STOP (R3'/R7 게이트)
    h = match(from_dict({"weight_pct": 0.025, "bw_pct": 0.6, "latency_us": 36.0}), rules)
    assert h is not None and h.label == "below_weight_gate" and h.is_stop, h

    # self-check 2: 비중≥5% fp32 matmul 텐서코어 off → TF32 (R5 최대 이득)
    h = match(from_dict({"weight_pct": 0.527, "compute_tput": 0.8,
                         "tensorcore_active": False, "latency_us": 100.0}), rules)
    assert h is not None and h.label == "fp32_no_tensorcore", h
    assert not h.is_stop, h

    # self-check 3: 텐서코어 이미 활성 + 느림 → STOP, 저정밀 무효 (R6 bf16 반증)
    h = match(from_dict({"weight_pct": 0.49, "tensorcore_active": True,
                         "latency_us": 100.0}), rules)
    assert h is not None and h.label == "tensorcore_saturated" and h.is_stop, h

    # self-check 4: 비중≥5% 메모리바운드 텐서코어 off → 융합 (R3' 약적중)
    h = match(from_dict({"weight_pct": 0.11, "bw_pct": 0.7, "compute_tput": 0.1,
                         "tensorcore_active": False, "latency_us": 36.0}), rules)
    assert h is not None and h.label in ("fp32_no_tensorcore", "memory_bound_fusable"), h
    # compute_tput 낮으면 fp32_no_tensorcore의 compute_tput>0 조건은 통과하나
    # priority 1(텐서코어)이 2(융합)보다 우선 → fp32_no_tensorcore 선택될 수 있음.
    # 순수 메모리바운드(compute_tput=0) 케이스:
    h = match(from_dict({"weight_pct": 0.11, "bw_pct": 0.7, "compute_tput": 0.0,
                         "tensorcore_active": False, "latency_us": 36.0}), rules)
    assert h is not None and h.label == "memory_bound_fusable", h

    # self-check 5: 아무 룰도 안 맞으면 None (작은 비중 0, 정상)
    h = match(from_dict({"weight_pct": 0.0, "bw_pct": 0.5, "load_eff": 0.9,
                         "occupancy": 0.9, "latency_us": 10.0}), rules)
    assert h is None, h

    # self-check 6 (a, nsys): GPU idle 큼 + 지배커널 없음 → launch_overhead
    # load_eff 기본 0.0이라 uncoalesced(p4)도 발화하나 launch_overhead(p2) 우선.
    h = match(from_dict({"launch_gap_pct": 0.4, "weight_pct": 0.2,
                         "load_eff": 0.9, "latency_us": 30.0}), rules)
    assert h is not None and h.label == "launch_overhead", h

    # self-check 7 (a, torch): attention 지배 → attention_dominant
    h = match(from_dict({"op_weight": 0.54, "op_name": "aten::scaled_dot_product_attention",
                         "weight_pct": 0.0, "latency_us": 100.0}), rules)
    assert h is not None and h.label == "attention_dominant", h

    # self-check 8 (a 가드): attention op지만 비중<40% → 발화 안 함 (오발화 방지)
    h = match(from_dict({"op_weight": 0.2, "op_name": "aten::sdpa",
                         "weight_pct": 0.0, "load_eff": 0.9, "latency_us": 10.0}), rules)
    assert h is None, h

    # ── self-check 9~12: 칩 컨텍스트 가드 (design 07) ──
    from signals import Context
    # fp32 matmul 신호 (self-check 2와 동일). 칩별로 발화가 갈려야 함.
    fp32_sig = {"weight_pct": 0.527, "compute_tput": 0.8,
                "tensorcore_active": False, "latency_us": 100.0}
    # 9: ctx 없음(=A100 가정) → 종전대로 fp32_no_tensorcore (회귀 0)
    h = match(from_dict(fp32_sig), seed_rules())
    assert h is not None and h.label == "fp32_no_tensorcore", h
    # 10: A100 명시 → 동일 (TF32 있음)
    h = match(from_dict(fp32_sig), seed_rules(), Context(chip="a100"))
    assert h is not None and h.label == "fp32_no_tensorcore", h
    # 11: T4 → TF32 없음 → fp32_no_tensorcore 차단. 같은 신호서 다음 적격 룰로 떨어짐.
    #     (memory_bound_fusable는 bw_pct>0.5 필요 — 여기선 없으니 None or 타 룰)
    h = match(from_dict(fp32_sig), seed_rules(), Context(chip="t4"))
    assert h is None or h.label != "fp32_no_tensorcore", h
    # 12: T4 + 메모리바운드 신호 → fp32 룰 막히고 memory_bound_fusable 발화 (진화 흡수 경로)
    h = match(from_dict({"weight_pct": 0.2, "bw_pct": 0.7, "compute_tput": 0.0,
                         "tensorcore_active": False, "latency_us": 50.0}),
              seed_rules(), Context(chip="t4"))
    assert h is not None and h.label == "memory_bound_fusable", h
    # 13: tensorcore_saturated도 T4선 STOP 부적용 (TF32≈bf16 근거가 TF32 칩 전제)
    h = match(from_dict({"weight_pct": 0.49, "tensorcore_active": True,
                         "latency_us": 100.0}), seed_rules(), Context(chip="t4"))
    assert h is None or h.label != "tensorcore_saturated", h

    print("rules.py self-check PASS")
