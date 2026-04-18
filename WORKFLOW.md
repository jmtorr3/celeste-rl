# Development Workflow

The project compares three approaches: **DQN** (pure RL), **Behavioral Cloning** (pure imitation), and **Hybrid** (DQN warm-started from TAS data). The central claim is that the hybrid outperforms both.

The loop within each approach: **train → evaluate → identify weakness → fix → new version**

---

## Project Roadmap

```
Phase 1 — Fix foundations (professor feedback)        ✓ DONE
Phase 2 — DQN iterations (v1 baseline → improved)     ✓ DONE — result: 0% completion, best height -3
Phase 3 — Behavioral Cloning (TAS data pipeline + BC model)   ← YOU ARE HERE
Phase 4 — Hybrid (DQN + TAS pre-loaded replay buffer)
Phase 5 — Final evaluation and comparison
```

---

## Phase 1 — Required Fixes (Professor Feedback) ✓ DONE

All applied to `src/environment.py`:
- ✓ Coordinate normalization: `player.x / 64 - 1` → `player.x / 128.0`
- ✓ 5×5 tile grid added to state (6 → 31 dims) via `game.tile_at(x, y)`
- ✓ `_get_obs_dim()` returns 31

---

## Phase 2 — DQN Versions ✓ DONE

**Result: 0% completion across all runs. Best height reached: -3 (5 pixels from exit). DQN model for final evaluation: `models/v3_r6_best.pt` (or best available from run 6).**

| Version | Script | Change |
|---------|--------|--------|
| v1 | `src/train.py` | Baseline DQN, 6-dim state |
| v2 | `src/train_v2.py` | Exploration bonuses (position visit counts) |
| v3 | `src/train_v3.py` | Professor fixes (31-dim state) + Dueling/Double DQN + progressive rewards + milestones |

### train_v3.py — always use `--run-id`

Each run saves `{run_id}_best.pt`, `{run_id}_checkpoint_ep{n}.pt`, `{run_id}_final.pt`. **Without a unique run ID, new runs overwrite old models.** A run-4 model reaching height=-3 was lost this way.

```bash
python src/train_v3.py --episodes 5000 --epsilon-decay 0.999990 --device cuda --run-id v3_r6
```

---

## Phase 3 — Behavioral Cloning

BC treats imitation as supervised classification: map `(state, action)` pairs from TAS recordings.

### TAS data pipeline

```bash
# 1. Clone TAS recordings (UniversalClassicTas — contains actual input files)
git clone https://github.com/CelesteClassic/UniversalClassicTas pyleste/UniversalClassicTas
# Note: https://github.com/CelesteClassic/tasdatabase is metadata only (frame counts etc.) — not the inputs

# 2. Export (s, a) pairs by replaying TAS through the emulator
python scripts/export_tas.py --output data/tas_transitions.pkl

# 3. Train BC model
python src/train_bc.py --data data/tas_transitions.pkl
```

### TAS file format

TAS files are plain text, comma-separated integers — one per frame, same bitmask format `set_btn_state()` already uses:

```
0,0,18,2,2,2,34,0,0,16,...
```

Parsing is trivial:
```python
inputs = [int(x) for x in open("tas_file.txt").read().split(",") if x.strip()]
```

`CelesteUtils.watch_inputs(p8, inputs)` already replays this format through the emulator, so `export_tas.py` is mostly a loop that calls `watch_inputs` and records `(state, action)` at each frame.

### BC notes
- Use **action-weighted cross-entropy loss** — TAS is heavily biased toward right/up, dash is rare
- Inject **Gaussian noise** (σ=0.02) on kinematic features (x, y, spd_x, spd_y) during training to reduce overfitting — do NOT add noise to `grace` or `djump`
- Expected: 30–60% success on in-distribution states, degrades off-trajectory

---

## Phase 4 — Hybrid (DQN + TAS Pre-loading)

Pre-fill the DQN replay buffer with expert TAS transitions before online training begins.

```python
# In train_hybrid.py, before the training loop:
for transition in tas_transitions:
    agent.buffer.push(*transition)
# Then run normal DQN training loop
```

Key experiment: reserve a fixed fraction of the buffer for expert data to prevent catastrophic forgetting as online transitions fill it.

---

## Quick Commands

```bash
# Train v3 (always use --run-id)
python src/train_v3.py --episodes 5000 --epsilon-decay 0.999990 --device cuda --run-id v3_rN

# Resume from checkpoint
python src/train_v3.py --resume models/v3_rN_checkpoint_ep500.pt --run-id v3_rN --device cuda

# Eval only
python src/train_v3.py --eval-only --model models/v3_rN_best.pt

# Watch
./watch.sh -m models/v3_rN_best.pt

# Evaluate
python scripts/evaluate.py --model models/v3_rN_best.pt --episodes 100 --baseline

# Google Drive sync — push after EVERY run before starting a new one
./sync_models.sh push
./sync_models.sh pull     # on new machine
./sync_models.sh status
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Success rate | % of episodes completing the level |
| Mean episode reward | Avg shaped reward (50-ep moving avg) |
| Best / mean height reached | Vertical progress (lower y = higher) |
| Death rate | Died vs. timed out |
| Steps to completion | On successful episodes only |

Final comparison: Random → DQN → BC → Hybrid → TAS (upper bound)

---

## Model Naming Convention

All v3+ models use a `--run-id` prefix to avoid overwrites. BC and Hybrid follow the same pattern.

| Model | Files |
|-------|-------|
| v1 (legacy) | `models/dqn_best.pt`, `models/dqn_final.pt` |
| v2 (legacy) | `models/model_v2_best.pt`, `models/model_v2_final.pt` |
| v3 run N | `models/v3_rN_best.pt`, `models/v3_rN_checkpoint_ep{n}.pt`, `models/v3_rN_final.pt` |
| BC | `models/bc_best.pt`, `models/bc_final.pt` |
| Hybrid | `models/hybrid_best.pt`, `models/hybrid_checkpoint_ep{n}.pt`, `models/hybrid_final.pt` |

All `models/*.pt` are gitignored — sync to Google Drive with `./sync_models.sh`. **Push after every run.**

---

## Google Drive Sync (rclone)

```bash
# First-time setup
brew install rclone
rclone config   # create a remote named "gdrive"
# Then edit REMOTE= in sync_models.sh to your Drive folder path
```
