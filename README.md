# GPU-Solver

An agentic GPU kernel-optimization loop where an **LLM agent evolves its own classification
rules from profiling measurements**. The evolution mechanism is proven with a controlled
ON/OFF ablation on real A100 hardware. Performance gain was measured to be null and attributed
to an environment limit — stated honestly as future work.

> This README is the methodology narrative. For internals see [`loop/README.md`](loop/README.md).

## One line

Prior art (CUDAMaster, arXiv 2603.07169) already implements the "deterministic rule label →
LLM rewrite → measure-verify" pipeline. **My contribution = the classification rule table itself
evolves from measurement feedback.** All 6 surveyed prior systems use static rules.

## What I proved (claimable)

### 1. Measurement-feedback rule-evolution meta-loop — mechanism proven

**Evidence (real A100, sigmoid, 14 rounds, `loop/run_gain_compare.py`):**
- Evolution **ON** = a wrong `fp32_no_tensorcore` rule fired 4 times, failed, was retired →
  auto-switched to `memory_bound_fusable`.
- Evolution **OFF** = the same fp32 rule misfired 6 times forever (fake signal).
- = "a wrong static rule gets retired by measurement and replaced by the right one" — closed on real GPU.

### 2. Self-validation via ablation (the core contribution)

The ON/OFF above is not a demo. It is an **ablation that isolates which component (evolution)
produces the effect.** The same variant queue was fed fairly to both tracks, so the difference
is controlled to the single variable of evolution presence/absence.

### 3. The loop reaches a faster kernel by measurement — gain layer first breakthrough

| Problem | Attempt | Result |
|---|---|---|
| **matmul (4096² fp32)** | loop R0(fp32)→`fp32_no_tensorcore` fires→R1(TF32) | ✅ **9.6ms→1.5ms = 6.4× gain** |
| sigmoid | `--latency` 12R, BLOCK variants | null — memory-bound ceiling (no headroom) |
| groupnorm | split-parallel (3 algorithms) | null — DRAM BW ceiling |
| llama | TF32 OFF vs ON | null — attention (flash) 54% dominates, matmul 23% not bottleneck |

**matmul: the loop autonomously reached a 6.4×-faster kernel via measure→hypothesis→rewrite.** First
demonstration of "measure → hypothesis → rewrite → faster kernel." sigmoid/groupnorm nulls are real
ceilings (memory-bound); llama is attention-dominated — not a loop defect, a problem property.

> ⚠️ **Measurement integrity:** matmul first read 95ms parity ("environment limit"), but the cause
> was a bug — `_profile_ncu` measured only 1 kernel (`--launch-count 1`). Summing all-kernel durations
> revealed the 6.4×, cross-checked by ncu kernel names (sgemm vs tensorop) and theoretical FLOP.

## What I did NOT prove

- ❌ **Evolution superiority ON>OFF.** On matmul the first fired rule is correct → no retire → ON≈OFF.
  Evolution's benefit is shown separately by sigmoid's misfire-retire. The gain axis and evolution axis
  are not demonstrated together on one problem.
- ❌ **Multi-problem generalization.** gain=matmul (1 problem), evolution=sigmoid (1 problem).
- ⚠️ **Two axes, separated:** loop improves (gain ✅ matmul) + evolution beats static (✅ sigmoid) =
  each single-problem. Both-on-one + multi-problem generalization = future work.
  - **Both-on-one confirmed structurally hard (by measurement).** A deliberately uncoalesced Triton matmul
    was tried to get "wrong rule → retire → right rule → gain," but the seed rules fit so well the first
    fired rule is always correct → no retire. Wrong-rule-fires-naturally = memory-bound (gain ceiling) vs
    right-rule = compute (no retire) = mutually exclusive. The limit itself was established by measurement.

## Repository layout

```
loop/                  # the optimization loop (separate concerns, self-checkable)
  signals.py           # profiling-signal extraction (bw_pct, tensorcore_active, ...)
  rules.py             # classification rule table (seed rules)
  evolver.py           # rule evolution: confidence ±1, retire, candidate proposal
  ledger.py            # round/decision history
  generator.py         # LLM kernel rewrite (callback for PoC; real generator behind API key)
  harness.py           # optimization round driver
  executor.py          # kernel run / measurement
  mailbox.py           # local side of git-mailbox async cmd/result channel
  watch.py             # Colab side of git-mailbox (polling, idempotent, fault-isolated)
  run_gain_compare.py  # evolution ON/OFF ablation (the mechanism proof)
  run_gain_hypcond.py  # hypothesis-conditional callback driver
  run_multiproblem.py  # multi-problem rule-firing observation
  selfcheck.py         # GPU-free local self-check
problems/{llama,sigmoid,groupnorm}/solve.py
colab_mailbox.ipynb    # Colab watch notebook (auth via Colab Secrets)
```

## Run

```bash
# GPU-free local self-check (logic only — no torch/ncu)
python3 loop/selfcheck.py

# evolution ON/OFF ablation (needs a live Colab A100 watch via git-mailbox)
python3 loop/run_gain_compare.py <problem>
```

The git-mailbox pattern (this repo ↔ a separate `gpu-mailbox` repo) carries async cmd/result
JSON between a local runner and a Colab `watch` process — no SSH tunnel, deployable with a single
repo-scope PAT.

**Full setup (mailbox repo, Colab watch, PAT via Secrets): [`docs/SETUP.md`](docs/SETUP.md).**

## Reproducibility

The gain-measurement infrastructure is ready — retry as-is once the environment is PoC-grade (1.4ms):
- `loop/run_gain_hypcond.py` = hypothesis-conditional callback driver (selects code by fired rule label).

## License

Personal portfolio / research prototype.
