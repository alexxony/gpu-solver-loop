# Development Method

This describes the process used to develop and validate the results in this repository (and its
sibling projects, compiler-thermal, hbm-build). It is a record of five rules followed during the
work, and what those rules caught.

## The five rules

**1. ON/OFF controlled comparison.** Within the same pool of variants, one condition is toggled
while everything else stays fixed. gpu-solver-loop toggles evolution on/off; compiler-thermal runs
the KernelBench ablation; hbm-build compares top-only vs. top+bottom cooling with the rest of the
setup held constant.
Evidence: `hbm_build/JOURNAL.md` 2026-07-19T22:29:53, `results/p4_report.md` §3-§6.

**2. Immediate ledger recording.** Every run's result is written to a structured file at the time
it happens, not reconstructed afterward from memory or a summary.
Evidence: `compiler_thermal/loop/JOURNAL.md` (second-resolution ISO 8601 timestamps throughout).

**3. Negative results reported as-is.** When a prior expectation is wrong, or a failure doesn't
resolve, it's written down rather than smoothed over, with a stated reason when it's carried
forward instead of fixed.
Evidence: `hbm_build/JOURNAL.md` 2026-07-21T01:21:28 (0.7616 FAIL carried forward, not hidden).

**4. Hypotheses documented before experiments; reversals reported as reversals.** A design document
states the expected outcome before the experiment runs. If the result comes back the opposite way,
that's recorded as a reversal, not quietly rewritten to match.
Evidence: `Compiler_Thermal/docs/11-p10-hotspot-deltat-design.md:194-199` (hypothesis: attenuation
expected) and `:252` (reversal recorded: amplification observed across all scenarios instead).

**5. Failure triage by reconstruction.** When a result fails a check, the comparison is
recomputed under a different axis or statistic. If the outcome changes, the original failure was a
comparison artifact. If it doesn't change, the failure is treated as a real gap and carried forward
rather than dismissed.
Evidence: `hbm_build/JOURNAL.md` 2026-07-20T23:50:08 (max-vs-avg axis mismatch found and
reconstructed as avg-vs-avg → PASS) and 2026-07-21T01:21:28 (same reconstruction applied to a
second series, result did not change → failure stood).

## What this process caught

- An energy-integration definition bug that structurally penalized fast kernels (fixed
  power-integration window biased against short runtimes).
- A sign error in a reporter script, found by cross-checking its output against the raw metric
  curve it was summarizing rather than trusting the reporter's own self-check.
- A comparison-axis artifact (max-vs-avg mismatch) that had produced a false FAIL; reconstructing
  the comparison on a consistent axis (avg-vs-avg) turned it into a PASS.
- A pre-registered hypothesis (attenuation) that was contradicted by the result (amplification,
  1.23-1.64x) — reported as a reversal instead of revised after the fact.
- An axis-mixing error in an uncommitted report draft (an average-based figure compared against a
  max-based one), caught by reconstructing the calculation from source rather than trusting the
  stated number.
- A duplicate work assignment, caught mid-execution by checking version-control history before
  proceeding, avoiding redundant work.

**What it did not catch:** one failure series (G4 B-series, 0.7616) was recomputed under the same
reconstruction method used elsewhere and did not resolve — it remains an open, unexplained gap,
carried forward rather than closed.

## Verifier isolation

Where a verification step is used, the verifier receives only raw artifacts (CSV, JSON, logs) —
not the implementer's summary or conversation history. This isn't specific to a particular tool;
it holds for structural reasons: a shared context accumulates low-signal execution log volume that
crowds out judgment, an implementer who produced an artifact is inclined to defend it, and an
implementer optimizes for "make it pass" while a verifier optimizes for "is this claim true" — the
two aren't the same objective. This pattern is documented in later projects in this stack
(compiler-thermal, hbm-build); it is not claimed for gpu-solver-loop, which has no equivalent
documented evidence.

## In this repo

gpu-solver-loop is the origin of rules 1 and 2 (ON/OFF ablation, immediate ledger recording) and
the reference engine (`rules.py`, `evolver.py`, `ledger.py`) that the other two projects hold
unchanged. It has no documented evidence for pre-registration (rule 4) or verifier isolation.
