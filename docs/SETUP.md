# Setup — running the loop on a real GPU

The optimization loop runs locally (CPU only — orchestration, rule evolution, LLM rewrite),
but kernel measurement needs a GPU. This repo uses a **git-mailbox** pattern: a local runner
and a Colab `watch` process exchange cmd/result JSON through a separate GitHub repo. No SSH
tunnel, no Drive mount — deployable with a single repo-scope PAT.

```
local runner  ──push REQ──▶  mailbox repo (GitHub)  ──pull──▶  Colab (A100 + watch.py)
local runner  ◀──pull RES──   mailbox repo (GitHub)  ◀──push──   Colab
```

## Prerequisites

- A GitHub Personal Access Token, **repo scope only**.
- A Google Colab account with GPU runtime (A100 used here; any CUDA GPU works for the mechanism).
- Local Python 3 (the local side needs no torch — it only does git + JSON + orchestration).

## 1. Create the mailbox repo

The mailbox is a **separate, empty repo** — kept apart from this code repo so hundreds of
transient cmd/result JSON commits don't pollute the code history. Make it **private**
(generated kernel code passes through it).

```bash
gh repo create <you>/gpu-mailbox --private
git clone https://github.com/<you>/gpu-mailbox.git ~/workspace/gpu-mailbox
mkdir -p ~/workspace/gpu-mailbox/{cmd,result,done}
touch ~/workspace/gpu-mailbox/{cmd,result,done}/.gitkeep
cd ~/workspace/gpu-mailbox && git add -A && git commit -m "init mailbox" && git push
```

## 2. Start the Colab watch process

1. Open `colab_mailbox.ipynb` in Colab (GitHub tab → this repo → the notebook).
2. **Store the PAT in Colab Secrets** (🔑 sidebar) as `GPU_MAILBOX_TOKEN` — never paste it in a
   cell. The notebook reads it via `userdata.get`, so it never appears in output or git history.
3. Run: cell 1 (`nvidia-smi`, confirm GPU) → clone cell → watch cell.
4. The watch cell loops `git pull` every ~5s with `max_iters=None` (runs until you stop it).
   Polling logs = it's alive.

> `watch.py` lives at `loop/watch.py`, not the repo root — the Colab cell does
> `cd .../loop` and adds `/loop` to `sys.path`.

## 3. Run from local

```bash
# GPU-free logic self-check first (no torch/ncu needed)
python3 loop/selfcheck.py

# evolution ON/OFF ablation — the mechanism proof (needs the watch alive)
python3 loop/run_gain_compare.py <problem>      # e.g. sigmoid

# single optimization step
python3 loop/run_gain_round.py <problem> <variant_solve.py>
```

The local driver pushes a REQ, the Colab watch compiles/measures and pushes a RES, the driver
pulls it back. Measurement signals (`bw_pct`, `tensorcore_active`, `latency_us`, ...) feed the
rule table; the evolver retires misfiring rules and switches to better ones across rounds.

## Notes

- **Two repos, two branches.** This code repo is on `master`; the mailbox repo is on `main`.
- **Clean the mailbox** between gain rounds: remove everything under `cmd/ result/ done/`
  (done markers have no extension), commit, push.
- **Code changes need a Colab runtime restart** — re-cloning alone serves the cached old module
  (`sys.modules`). Re-clone only is enough if loop code is unchanged.
- The PoC measurement environment (~1.4ms) differs from this git-mailbox Colab (~24ms on llama).
  The mechanism is environment-independent; performance-gain measurement is not — see the README.
