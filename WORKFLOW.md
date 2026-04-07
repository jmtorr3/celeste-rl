# Development Workflow

The project compares three approaches: **DQN** (pure RL), **Behavioral Cloning** (pure imitation), and **Hybrid** (DQN warm-started from TAS data). The central claim is that the hybrid outperforms both.

The loop within each approach: **train → evaluate → identify weakness → fix → new version**

---

## Project Roadmap

```
Phase 1 — Fix foundations (professor feedback)        ← YOU ARE HERE
Phase 2 — DQN iterations (v1 baseline → improved)
Phase 3 — Behavioral Cloning (TAS data pipeline + BC model)
Phase 4 — Hybrid (DQN + TAS pre-loaded replay buffer)
Phase 5 — Final evaluation and comparison
```

---

## Phase 1 — Required Fixes (Professor Feedback)

These must be done before further training. See `docs/professor_feedback_fixes.md` for full details.

- [ ] **Fix coordinate normalization** — `src/environment.py` `_get_obs()` lines 164–165
      Change `player.x / 64 - 1` → `player.x / 128.0`, same for y
      Also fix in `src/train_v2.py` `CelesteEnvV2._get_obs()`

- [ ] **Add tile grid to state** — expand state from 6 → 31 dims
      Append flattened 5×5 tile grid centered on player
      `0.0` = empty, `1.0` = solid, `-1.0` = hazard/spike
      Files: `src/environment.py`, `src/train_v2.py`
      ✓ `game.tile_at(x, y)` already exists in `pyleste/Carts/Celeste.py:568` — no workaround needed

- [ ] **Update `_get_obs_dim()`** — return 31 after tile grid added

---

## Phase 2 — DQN Versions

| Version | Script | Best Model | Change |
|---------|--------|------------|--------|
| v1 | `src/train.py` | `models/dqn_best.pt` | Baseline DQN |
| v2 | `src/train_v2.py` | `models/model_v2_best.pt` | Exploration bonuses (position visit counts) |
| v3 | `src/train_v3.py` | `models/v3_best.pt` | Apply professor fixes (31-dim state, fixed coords) |
| v4+ | `src/train_v4.py` | `models/v4_best.pt` | Dueling DQN, Double DQN, curriculum, etc. |

### Creating a new DQN version

```bash
cp src/train_v2.py src/train_v3.py
# edit src/train_v3.py — make one change, update save paths to models/v3_*
./train.sh -v 3
./watch.sh -v 3
```

### DQN improvement ideas (in rough priority)

1. **Apply professor fixes** (tile grid + coord fix) → v3
2. **Dueling DQN** — swap `DQN` → `DuelingDQN` in `agent.py` (already implemented in `src/network.py`)
3. **Double DQN** — use policy net to select action, target net to evaluate it (~5 line change in `agent.py:update()`)
4. **Curriculum** — train on room 0, load checkpoint, continue on rooms 1→2→3
5. **Prioritized replay** — replay high TD-error transitions more often
6. **Tune epsilon** — try decay 0.999 (faster) vs 0.9998 (more exploration)

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
# Train
./train.sh -v 1                          # DQN baseline
./train.sh -v 2 -e 5000                  # v2 with more episodes
./train.sh -v 3                          # new version
./train.sh -v 3 -m models/v3_checkpoint.pt  # resume

# Watch
./watch.sh -v 1                          # v1 best model
./watch.sh -v 2 -e 5 -d 0.05            # v2, slow
./watch.sh -m models/v3_best.pt          # any model by path

# Evaluate
python scripts/evaluate.py --model models/v3_best.pt --episodes 100 --baseline

# Google Drive sync
./sync_models.sh push     # after training — upload models/
./sync_models.sh pull     # on new machine — download models/
./sync_models.sh status   # check what's out of sync
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

| Version | Best | Checkpoint | Final |
|---------|------|------------|-------|
| v1 (legacy) | `models/dqn_best.pt` | `models/dqn_checkpoint.pt` | `models/dqn_final.pt` |
| v2 (legacy) | `models/model_v2_best.pt` | — | `models/model_v2_final.pt` |
| v3+ | `models/v3_best.pt` | `models/v3_checkpoint.pt` | `models/v3_final.pt` |
| BC | `models/bc_best.pt` | — | `models/bc_final.pt` |
| Hybrid | `models/hybrid_best.pt` | `models/hybrid_checkpoint.pt` | `models/hybrid_final.pt` |

All `models/*.pt` are gitignored — sync to Google Drive with `./sync_models.sh`.

---

## Google Drive Sync (rclone)

```bash
# First-time setup
brew install rclone
rclone config   # create a remote named "gdrive"
# Then edit REMOTE= in sync_models.sh to your Drive folder path
```
