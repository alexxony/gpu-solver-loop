"""Trace Parser (★내 코드) — 원시 ncu/nsys/torch-profiler 출력 → 정규화 신호 dict.

설계: design spec §3.1. "무슨 신호를 뽑을지" 선택 = perf 이해 증거.
GPU 불필요 — CSV/dict in, 정규화 dict out. Colab 프로파일러가 뱉은 걸 여기서 정규화.

세 소스 역할 분담 (판정은 룰이, profiler는 신호 소스일 뿐):
- ncu  = 커널 내부 (왜 느린가): occupancy/load_eff/bw_pct/tensorcore_active.
- nsys = 타임라인 (어디가 병목): weight_pct(커널 비중), launch_gap(런치 오버헤드).
- torch profiler = 연산자 귀속 (무엇인가): op_name/op_weight/op_shape (싼 1차 스크리닝).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Signal:
    """정규화 신호. 룰이 읽는 4입력 중 1번(프로파일 신호)."""
    occupancy: float = 0.0        # warps_active / max
    reg_per_thread: int = 0
    l2_hit: float = 0.0
    load_eff: float = 0.0         # global_load_efficiency (합착도)
    stall_reason: str = ""        # sync / mem / none
    bw_pct: float = 0.0           # achieved DRAM throughput %  (0~1)
    compute_tput: float = 0.0     # achieved compute %          (0~1)
    latency_us: float = 0.0       # 커널 측정 시간 (느림 판정용)
    # PoC R3~R7 trajectory가 요구한 신호 (per-kernel 메트릭만으론 부족)
    weight_pct: float = 0.0       # 이 커널 / 전체 latency 비중 (0~1). ≥5% 게이트 핵심.
    tensorcore_active: bool = False  # 텐서코어 경로 사용? (sgemm vs TC-gemm 구분)
    # nsys 타임라인 신호 (커널 내부 아닌 "어디가 병목")
    launch_gap_pct: float = 0.0   # 커널 간 idle / 전체 (0~1). 높으면 launch 오버헤드·CPU 병목
    # torch profiler 신호 (연산자 의미 귀속 — 싼 1차 스크리닝)
    op_name: str = ""             # 지배 연산자 (aten::mm / sdpa / layer_norm ...)
    op_weight: float = 0.0        # 그 연산자 / 전체 self CUDA time (0~1)
    op_shape: str = ""            # 입력 텐서 모양 (record_shapes=True). "큰 GEMM vs 작은" 룰용

    # 룰 람다가 짧게 쓰도록 alias (spec §3.2 표기와 일치)
    @property
    def occ(self) -> float: return self.occupancy
    @property
    def reg(self) -> int: return self.reg_per_thread


# ncu --csv 컬럼명 → Signal 필드 매핑 (실제 ncu 메트릭 ID)
NCU_METRIC_MAP = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy",
    "launch__registers_per_thread": "reg_per_thread",
    "lts__t_sector_hit_rate.pct": "l2_hit",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "load_eff",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "bw_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_tput",
    "gpu__time_duration.sum": "latency_us",
    # 텐서코어 파이프 활성 사이클 (>0 = TC 경로). bf16/TF32 무효 판정용.
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active": "_tc_pct",
}
# weight_pct = 커널/전체 비중, ncu 단일 메트릭 아님 → parse_ncu_rows 밖에서
# 전체 합 대비로 주입 (harness가 채움). tensorcore_active = _tc_pct > 0.


def _to_fraction(field: str, raw: float) -> float:
    """pct 메트릭은 0~100으로 옴 → 0~1 정규화. 카운트/시간은 그대로."""
    pct_fields = {"occupancy", "l2_hit", "load_eff", "bw_pct", "compute_tput"}
    return raw / 100.0 if field in pct_fields else raw


def parse_ncu_rows(rows: list[dict]) -> Signal:
    """ncu --csv 파싱 결과(행 리스트, 각 {metric_name, value}) → Signal.

    한 커널의 여러 메트릭 행이 들어온다고 가정. 모르는 메트릭은 무시.
    """
    sig = Signal()
    for r in rows:
        name = r.get("Metric Name") or r.get("metric_name") or r.get("name")
        val = r.get("Metric Value") or r.get("value")
        if name is None or val is None:
            continue
        field = NCU_METRIC_MAP.get(name)
        if field is None:
            continue
        try:
            raw = float(str(val).replace(",", ""))
        except ValueError:
            continue
        if field == "_tc_pct":           # 텐서코어 활성 % → bool
            sig.tensorcore_active = raw > 0.0
        else:
            setattr(sig, field, _to_fraction(field, raw))
    return sig


def parse_nsys_rows(rows: list[dict], sig: Signal | None = None) -> Signal:
    """nsys gpukernsum/gputrace 요약 행 → 타임라인 신호 주입.

    각 행 {name, total_ns 또는 duration_ns}. 커널 duration 합 + 전체 wall 대비로
    weight_pct(지배 커널 비중), launch_gap_pct(idle 비율) 계산.
    ncu가 못 주는 "어디가 병목" 신호. ncu 결과 위에 덮어쓰기 가능 (sig 전달).

    rows 형식: [{"name": str, "dur_ns": float}, ...] + 선택 {"wall_ns": float}.
    wall_ns 없으면 커널 합을 wall로 간주(gap=0).
    """
    sig = sig or Signal()
    kernels = [(r.get("name", ""), float(r.get("dur_ns") or r.get("total_ns") or 0))
               for r in rows if (r.get("dur_ns") or r.get("total_ns"))]
    if not kernels:
        return sig
    kernel_sum = sum(d for _, d in kernels)
    wall = next((float(r["wall_ns"]) for r in rows if r.get("wall_ns")), kernel_sum)
    if wall > 0:
        top_dur = max(d for _, d in kernels)
        sig.weight_pct = top_dur / wall                       # 지배 커널 비중
        sig.launch_gap_pct = max(0.0, (wall - kernel_sum) / wall)  # GPU idle 비율
    return sig


def parse_torch_profiler(rows: list[dict], sig: Signal | None = None) -> Signal:
    """torch.profiler key_averages() → 연산자 의미 귀속 신호.

    각 행 {key/name, self_cuda_us 또는 cuda_us, input_shapes(선택)}.
    지배 연산자(self CUDA time 최대) 1개를 op_name/op_weight/op_shape로 주입.
    싼 1차 스크리닝: "attention vs matmul vs norm"을 ncu 없이 의미 레벨로 라벨.
    """
    sig = sig or Signal()
    ops = []
    for r in rows:
        name = r.get("key") or r.get("name") or ""
        cu = r.get("self_cuda_us")
        if cu is None:
            cu = r.get("cuda_us") or r.get("self_cuda_time_total")
        if name and cu is not None:
            try:
                ops.append((name, float(cu), str(r.get("input_shapes", ""))))
            except (ValueError, TypeError):
                continue
    if not ops:
        return sig
    total = sum(c for _, c, _ in ops)
    name, cu, shape = max(ops, key=lambda x: x[1])
    sig.op_name = name
    sig.op_weight = cu / total if total > 0 else 0.0
    sig.op_shape = shape
    return sig


def from_dict(d: dict) -> Signal:
    """이미 정규화된 dict (테스트/수동 입력)에서 Signal 생성."""
    return Signal(**{k: v for k, v in d.items() if k in Signal.__annotations__})


# ── 환경 컨텍스트 (측정 아닌 사전 사실) — design [[07-chip-lang-context-design]] ──
# Signal=사후 측정 vs Context=사전 환경. 칩 TF32 가능여부는 신호로 안 나옴 → 1급 필요.
# 황금규칙: 사용자/탐지가 Context(무대)만 채움, 최적 룰은 측정→진화가 발견 (룰 손수정 금지).

# 칩 → capability: NVIDIA 공개 스펙, 사람이 1회 박음 (고정 사실, 추론 아님).
CHIP_CAPS = {
    "a100": {"tf32": True,  "tc_gen": 3, "bf16": True},
    "h100": {"tf32": True,  "tc_gen": 4, "bf16": True, "fp8": True},
    "t4":   {"tf32": False, "tc_gen": 1, "bf16": False},  # Turing — TF32 없음
    "v100": {"tf32": False, "tc_gen": 1, "bf16": False},  # Volta — fp16 TC만
}


@dataclass
class Context:
    """측정 아닌 환경 (커널 실행 *전* 이미 앎). 룰이 cond 가드로 *읽음*.

    chip = nvidia-smi/torch.cuda.get_device_capability 자동탐지 가능.
    lang = 1급 아님 (Mojo 환경 0·파일단위 입도 불일치). 문법 차이는 측정 신호가 흡수
           (Triton tl.dot TF32 자동 → tensorcore_active=True로 관측). 기록용으로만 보관.
    """
    chip: str = ""    # "a100" | "h100" | "t4" | "v100"  (빈값 = 미지 = 가드 통과)
    lang: str = ""    # "triton" | "cuda" | "mojo" | "torch"  (서술/기록용)

    def cap(self, key: str) -> bool:
        """룰의 chip_cap 요구를 이 칩이 만족하나. 통과=True.

        - key="" (칩 무관 룰): 항상 True (가드 없음).
        - 칩 미지/미등록: True (보수적 통과 = 현 A100 동작 보존, 회귀 0).
        - 칩 명시 + 그 칩이 능력 없음: False (헛가설 차단 — T4서 TF32 룰 등).
        """
        if not key:
            return True  # 가드 없는 룰 = 항상 통과
        if not self.chip or self.chip not in CHIP_CAPS:
            return True  # 미지 칩 = 보수적으로 통과 (회귀 0)
        return bool(CHIP_CAPS[self.chip].get(key, False))


# 칩 미지 기본 컨텍스트 = 모든 가드 통과 = 종전 동작. match(ctx=None)이 이걸 씀.
DEFAULT_CTX = Context()


def detect_chip(cc: tuple[int, int] | None = None, name: str = "") -> str:
    """compute capability 튜플 or 디바이스명 → 칩 키. watch가 채워 Context 생성.

    cc 예: (8,0)=A100, (9,0)=H100, (7,5)=T4, (7,0)=V100.
    GPU 없는 로컬선 호출 안 됨 — Colab watch가 torch.cuda로 채움.
    """
    CC_MAP = {(8, 0): "a100", (9, 0): "h100", (7, 5): "t4", (7, 0): "v100"}
    if cc and cc in CC_MAP:
        return CC_MAP[cc]
    n = name.lower()
    for key in CHIP_CAPS:
        if key in n:
            return key
    return ""  # 미지 = 가드 통과


if __name__ == "__main__":
    # self-check: ncu 행 파싱 + pct 정규화
    rows = [
        {"Metric Name": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
         "Metric Value": "83.0"},
        {"Metric Name": "sm__warps_active.avg.pct_of_peak_sustained_active",
         "Metric Value": "81.0"},
        {"Metric Name": "launch__registers_per_thread", "Metric Value": "96"},
        {"Metric Name": "unknown__metric", "Metric Value": "999"},  # 무시돼야
    ]
    s = parse_ncu_rows(rows)
    assert abs(s.bw_pct - 0.83) < 1e-9, s.bw_pct
    assert abs(s.occupancy - 0.81) < 1e-9, s.occupancy
    assert s.reg_per_thread == 96
    assert s.occ == s.occupancy and s.reg == s.reg_per_thread  # alias
    # from_dict 라운드트립
    s2 = from_dict({"bw_pct": 0.85, "compute_tput": 0.15, "load_eff": 1.0})
    assert s2.bw_pct == 0.85 and s2.load_eff == 1.0

    # nsys: 지배 커널 80ns/100 wall = weight 0.8, idle (100-90)/100 = gap 0.1
    ns = parse_nsys_rows([
        {"name": "gemm", "dur_ns": 80},
        {"name": "elementwise", "dur_ns": 10},
        {"wall_ns": 100},
    ])
    assert abs(ns.weight_pct - 0.8) < 1e-9, ns.weight_pct
    assert abs(ns.launch_gap_pct - 0.1) < 1e-9, ns.launch_gap_pct
    # wall 없으면 gap=0, weight=top/sum
    ns2 = parse_nsys_rows([{"name": "a", "dur_ns": 30}, {"name": "b", "dur_ns": 10}])
    assert abs(ns2.weight_pct - 0.75) < 1e-9 and ns2.launch_gap_pct == 0.0

    # torch profiler: sdpa가 self CUDA 최대 → op_name=sdpa, weight=60/100
    tp = parse_torch_profiler([
        {"key": "aten::scaled_dot_product_attention", "self_cuda_us": 60,
         "input_shapes": "[[1,8,2048,64]]"},
        {"key": "aten::mm", "self_cuda_us": 40},
    ])
    assert tp.op_name == "aten::scaled_dot_product_attention", tp.op_name
    assert abs(tp.op_weight - 0.6) < 1e-9, tp.op_weight
    assert tp.op_shape == "[[1,8,2048,64]]"
    # ncu Signal 위에 nsys/torch 덮어쓰기 합성 (한 Signal에 3소스 누적)
    merged = parse_torch_profiler(
        [{"key": "aten::mm", "self_cuda_us": 5}],
        parse_nsys_rows([{"name": "k", "dur_ns": 9}, {"wall_ns": 10}], s))
    assert merged.bw_pct == 0.83          # ncu 신호 보존
    assert abs(merged.weight_pct - 0.9) < 1e-9   # nsys 주입
    assert merged.op_name == "aten::mm"   # torch 주입

    # Context: 칩 미지 → 모든 cap 통과 (회귀 0). A100 → tf32 True. T4 → tf32 False.
    assert Context().cap("tf32") is True            # 빈 칩 = 통과
    assert Context(chip="zzz").cap("tf32") is True   # 미등록 칩 = 통과
    assert Context(chip="a100").cap("tf32") is True
    assert Context(chip="t4").cap("tf32") is False   # Turing = TF32 없음
    assert Context(chip="v100").cap("bf16") is False
    assert DEFAULT_CTX.cap("tf32") is True
    # detect_chip: cc 튜플 우선, 없으면 이름 매칭
    assert detect_chip((8, 0)) == "a100"
    assert detect_chip((7, 5)) == "t4"
    assert detect_chip(None, "NVIDIA A100-SXM4-40GB") == "a100"
    assert detect_chip((6, 1)) == ""                 # 미지 cc = 빈값
    print("signals.py self-check PASS")
