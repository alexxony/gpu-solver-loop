"""Trace Parser (★내 코드) — 원시 ncu/nsys 출력 → 정규화 신호 dict.

설계: design spec §3.1. "무슨 신호를 뽑을지" 선택 = perf 이해 증거.
GPU 불필요 — CSV/dict in, 정규화 dict out. Colab ncu가 뱉은 걸 여기서 정규화.
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


def from_dict(d: dict) -> Signal:
    """이미 정규화된 dict (테스트/수동 입력)에서 Signal 생성."""
    return Signal(**{k: v for k, v in d.items() if k in Signal.__annotations__})


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
    print("signals.py self-check PASS")
