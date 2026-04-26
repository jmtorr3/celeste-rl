# celeste-rl

Deep reinforcement learning agents (DQN, BC, Hybrid, Curriculum) trained to play Celeste Classic, built on the Pyleste emulator.

**Current best result:** v3_r9 — DQN with semantic tile encoding, **64% peak completion rate** on room 0 (1139/5000 episodes completed).

## Project structure

```
celeste-rl/
├── README.md
├── requirements.txt
├── train.sh / watch.sh           # convenience wrappers
│
├── pyleste/                      # Pyleste emulator (vendored)
│
├── src/                          # RL code
│   ├── environment.py            # CelesteEnv — Gymnasium-style wrapper
│   ├── network.py                # DQN, DuelingDQN
│   ├── agent.py                  # DQNAgent + replay buffer
│   ├── train.py                  # vanilla DQN (room 0)
│   ├── train_v2.py               # +exploration bonus
│   ├── train_v3.py               # +DuelingDQN, semantic tiles  ← current best
│   ├── train_hybrid.py           # BC warm-start + DQN
│   ├── train_curriculum.py       # spawn-position curriculum
│   └── utils/
│       ├── paths.py              # run_dir(run_id) helper
│       └── plot.py               # shared training-curve plot
│
├── scripts/
│   ├── watch_agent.py            # visualize a trained agent
│   └── evaluate.py               # measure success rate over N episodes
│
├── runs/{run_id}/                # all artifacts of one training run
│   ├── best.pt
│   ├── final.pt
│   ├── checkpoint_ep{N}.pt
│   ├── {run_id}_training.pkl
│   └── {run_id}_curve.png
│
├── data/                         # TAS expert demonstrations
├── models/                       # legacy checkpoints (pre-runs/ layout)
└── docs/                         # research artifacts (curves, eval results)
```

## Setup

### Local (Mac / Linux laptop)

```bash
git clone https://github.com/jmtorr3/celeste-rl.git
cd celeste-rl
python3 -m venv celeste-venv
source celeste-venv/bin/activate
pip install -r requirements.txt
```

### Cluster (Blackwell GB10)

```bash
ssh blackwell
cd ~/celeste-rl
source celeste-venv/bin/activate
git pull                          # always pull before training
```

## How it works

### Environment (`src/environment.py`)

87-dimensional state per frame:
- Player position (x/128, y/128) normalized to [0, 1]
- Player velocity (spd.x/4, spd.y/4)
- Grace frames (coyote time, /6)
- Dash availability (0 or 1)
- **9×9 semantic tile grid** centered on the player. Each tile maps to one of 5 classes:
  - `-2.0` out-of-bounds (sentinel — distinguishes "wall ends here" from "wall continues")
  - `-1.0` spike (death)
  - `0.0` empty / air
  - `0.5` other (decoration, fruit)
  - `1.0` solid (landable platform)

15 discrete actions (subset of full PICO-8 button combinations):
- Idle, Left, Right
- Jump, Jump+Left, Jump+Right
- Dash in 8 directions

### Reward (`src/environment.py:_compute_reward`)

Progressive height bonus + 15-step milestone ramp covering y=50 down to y=-5. Spike-deaths give -5.0, level completion gives +500.0. Stuck-frame penalty after 30 frames of no movement.

### Networks (`src/network.py`)

- **DQN** — plain MLP `Input → 256 → 256 → 128 → 15`
- **DuelingDQN** — shared features, separate value + advantage streams, combined as `Q = V + (A - mean(A))`. Used by v3, hybrid, and curriculum trainers.

## Train an agent

Each run gets its own folder under `runs/{run_id}/`. Always pick a fresh `--run-id` so you don't overwrite a previous run.

### Vanilla DQN with semantic tiles (current best baseline)

```bash
python src/train_v3.py --run-id v3_r10 --device cuda \
  --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
```

### Hybrid DQN + Behavioral Cloning (BC warm-start)

```bash
python src/train_hybrid.py --run-id hybrid_r2 --device cuda \
  --episodes 3000 --epsilon-decay 0.999990 --epsilon-end 0.05 \
  --data data/tas_transitions.pkl --expert-fraction 0.20
```

⚠️ TAS data needs regeneration with the new 87-dim state — see open task in `final_log.md`.

### Curriculum learning (spawn at increasing y values)

```bash
python src/train_curriculum.py --run-id curriculum_r5 --device cuda
```

### Resume from a checkpoint

```bash
python src/train_v3.py --run-id v3_r10b --device cuda --resume runs/v3_r10/checkpoint_ep2000.pt
```

## Watch a trained agent

```bash
./watch.sh -i v3_r9                     # loads runs/v3_r9/best.pt
./watch.sh -i v3_r9 -e 10 -d 0.05       # 10 episodes, slower playback
./watch.sh -m runs/v3_r9/checkpoint_ep1950.pt   # specific checkpoint
./watch.sh -m models/v3_r8_checkpoint_ep3000.pt # legacy path
```

⚠️ `watch_agent.py` runs with ε=0.10 by default. Many of our checkpoints only complete the level *stochastically* — pure greedy play (ε=0) often gets stuck. If you want deterministic play, edit `scripts/watch_agent.py:46`.

## Evaluate a checkpoint

```bash
# By run id (loads runs/{id}/best.pt)
python scripts/evaluate.py --run-id v3_r9 --epsilon 0.05 --dueling --episodes 100

# By explicit checkpoint path
python scripts/evaluate.py --model runs/v3_r9/checkpoint_ep1950.pt \
  --epsilon 0.05 --dueling --episodes 100

# Random-action baseline (no model)
python scripts/evaluate.py --baseline-only --episodes 100
```

`--epsilon 0.05` is usually what you want — matches training-time noise, gives an honest "real-world performance" number. `--epsilon 0` is pure greedy and often misleading because of the `best.pt` saving bug (see `final_log.md`).

## Run-id conventions

To keep the comparison clean across teammates, please use these prefixes:

| Prefix | Algorithm | Example |
|---|---|---|
| `v3_rN` | DQN with semantic tiles | `v3_r10` |
| `hybrid_rN` | BC + DQN | `hybrid_r2` |
| `bc_rN` | Pure behavioral cloning | `bc_r2` |
| `curriculum_rN` | Spawn-curriculum | `curriculum_r5` |

Increment `N` for each new run. Don't reuse run IDs.

## What to do next (open tasks)

1. **Backup `runs/v3_r9/`** — the headline result. Copy `best.pt` + `checkpoint_ep1950.pt` somewhere safe.
2. **Re-eval `runs/v3_r9/checkpoint_ep1950.pt` with ε=0.05** — get the real deterministic-ish completion rate.
3. **Fix `best.pt` save logic** — currently saves on first y=-4, should save on best rolling completion rate.
4. **Regenerate TAS data with new 87-dim state** — required before hybrid_r2 / bc_r2 will work.
5. **Run hybrid_r2** — expected to beat v3_r9.
6. **Run bc_r2** — sets the imitation-only floor.
7. **Run curriculum_r5** — completes the comparison matrix.

## Things to watch out for (gotchas)

- **The `< -8` completion threshold is wrong.** The real exit fires at `player.y < -4` strictly. We've fixed this everywhere except: any old training pickles will report 0% completion if you re-derive stats from height arrays. Use `info['completed']` instead.
- **`best.pt` is misleading.** It's saved when `info['max_height'] < best_height` first fires — usually a one-off lucky episode, not the trained policy. Eval `checkpoint_ep{N}.pt` instead. (Open task to fix the save logic — see open tasks.)
- **Old checkpoints incompatible.** State went from 31-dim → 87-dim and reward magnitudes increased. Pre-v3_r9 checkpoints can't be resumed or evaluated.
- **Always activate the venv before training.** `source celeste-venv/bin/activate`. The shebang in scripts won't pick up the right Python on the cluster.

## References

- [Pyleste](https://github.com/CelesteClassic/Pyleste) — Python Celeste Classic emulator
- [Celeste Classic](https://celesteclassic.github.io/) — original game
- [TAS database](https://celesteclassic.github.io/tasdatabase/classic/) — expert demonstration source
- [DQN paper](https://arxiv.org/abs/1312.5602)
- [Dueling DQN paper](https://arxiv.org/abs/1511.06581)
