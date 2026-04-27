# celeste-rl

A final project for VT CS 4824 (Machine Learning, Spring 2026) by Akhil Madipalli, Maryam Malik, and Jorge Manuel Torre.

We trained reinforcement learning agents to play room 0 of [Celeste Classic](https://celesteclassic.github.io/) — the first level of a tiny PICO-8 precision platformer — and compared five methods against each other: a random baseline, plain DQN, Dueling DQN with a curiosity bonus, Behavioral Cloning from a TAS, Hybrid (BC warm-start + DQN), and a 6-stage spawn curriculum.

The headline finding was that **state representation mattered more than algorithm choice**. Replacing raw PICO-8 tile IDs (0–255) with a 5-class semantic encoding (air / solid / spike / out-of-bounds / other) raised plain DQN's completion rate from about 8% to roughly 60% with no other change. None of the algorithmic enhancements we tried — Dueling, curiosity, BC, hybrid, or curriculum — beat plain DQN with the perception fix.

This README covers how the repo is laid out, how to set it up, and how to reproduce or extend our results. For the full debugging story — every run, every dead end, every bug we found along the way — see [`DEVLOG.md`](DEVLOG.md).

## Repo layout

```
celeste-rl/
├── pyleste/                 # vendored Pyleste — Python PICO-8 emulator
├── src/
│   ├── environment.py       # CelesteEnv — Gymnasium-style wrapper, reward shaping
│   ├── network.py           # DQN, DuelingDQN
│   ├── agent.py             # DQNAgent + replay buffer
│   ├── train.py             # plain DQN
│   ├── train_v3.py          # Dueling DQN + curiosity bonus
│   ├── train_bc.py          # behavioral cloning
│   ├── train_hybrid.py      # BC warm-start + DQN
│   ├── train_curriculum.py  # spawn-position curriculum
│   └── utils/               # plot helpers, run-folder paths
├── scripts/
│   ├── evaluate.py          # measure completion rate over N episodes
│   ├── compare_all.py       # batch-evaluate every model + draw comparison plots
│   ├── watch_agent.py       # watch an agent play in the terminal
│   ├── record_gif.py        # save a successful run as an animated GIF
│   └── export_tas.py        # convert .tas replays into (state, action) pairs for BC
├── runs/                    # per-run artifacts (gitignored)
├── data/                    # TAS expert demonstrations
└── docs/comparison/         # paper-ready figures + tables
```

Each training run writes to `runs/{run_id}/` — checkpoints (`best.pt`, `final.pt`, `checkpoint_ep{N}.pt`), the training history pickle, and a curve PNG. Use a fresh `--run-id` per run so you don't overwrite anything.

## Setup

```bash
git clone https://github.com/jmtorr3/celeste-rl.git
cd celeste-rl
python3 -m venv celeste-venv
source celeste-venv/bin/activate
pip install -r requirements.txt
```

Training in a reasonable amount of time needs a CUDA GPU. We trained on a Blackwell GB10 cluster; a T4 or better should work.

## How the environment works

The agent sees an 87-dimensional state vector each frame: six scalar features (player x and y normalized to [0, 1], x and y velocity, coyote-time grace frames, and dash availability), plus a 9×9 grid of tiles centered on the player. Tiles aren't raw IDs — each is mapped to one of five semantic classes:

| Class | Value | Meaning |
|---|---|---|
| out-of-bounds | -2.0 | sentinel — distinguishes "wall ends here" from "wall continues" |
| spike | -1.0 | death tiles (PICO-8 IDs 17, 27, 43, 59) |
| air | 0.0 | passable empty space |
| other | 0.5 | decoration, fruit, anything non-deadly and non-solid |
| solid | 1.0 | landable platform or wall |

This was the single biggest improvement in the project. With raw tile IDs, the agent had no way to tell a spike (ID 17) apart from a decoration (ID 18) — they're numerically adjacent. The semantic classes give it that information directly.

The action space is 15 discrete actions: idle, left, right, jump (alone, +left, +right), and dash in eight directions. Reward is dominated by a 15-step milestone ramp from y=50 down to y=−5, plus a small movement bonus, a stuck-frame penalty, and terminal rewards (+500 for completing the room, −5 for dying).

The room is complete when the player crosses `y < -4`, which is what Pyleste itself uses to trigger the next-room transition.

## Training

Pick a fresh run ID, then launch one of the trainers. All four take the same epsilon flags. Five thousand episodes is the standard budget — that's what every method in the comparison table was given.

**Plain DQN** (the winner — recommend this as the default):

```bash
python src/train.py --run-id dqn_r2 --device cuda \
  --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
```

**Dueling DQN with curiosity bonus** (works but underperforms plain DQN):

```bash
python src/train_v3.py --run-id v3_r10 --device cuda \
  --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
```

**Behavioral Cloning** (no env loop, fast — ~5 minutes):

```bash
python src/train_bc.py --run-id bc_r3 --device cuda --epochs 200
```

**Hybrid (BC warm-start + DQN)**:

```bash
python src/train_hybrid.py --run-id hybrid_r3 --device cuda \
  --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
```

Note: our hybrid implementation records expert transitions with `reward = 0`, which polluted the Q-learning bootstrap target and gave us 0% completion. A correct version would replay the TAS through the env to capture real rewards. We didn't fix this before the deadline; if you do, that's a meaningful contribution.

**Curriculum learning** (6 stages — spawns at increasing y, agent advances when 50% of recent episodes complete):

```bash
python src/train_curriculum.py --run-id curriculum_r6 --device cuda
```

To resume a DQN run from a checkpoint: pass `--resume runs/<run_id>/checkpoint_epN.pt`. Note that resuming resets epsilon to 1.0 and wipes the replay buffer, which often causes catastrophic forgetting — check the agent code if you need to preserve those.

## Evaluating a trained model

`scripts/evaluate.py` runs N episodes and reports completion / death / timeout rates. Architecture (plain DQN vs Dueling DQN) is auto-detected from the checkpoint.

```bash
# By run ID — loads runs/{id}/best.pt
python scripts/evaluate.py --run-id dqn_r1 --epsilon 0.05 --episodes 100

# Or point at a specific checkpoint
python scripts/evaluate.py --model runs/dqn_r1/checkpoint_ep5000.pt \
  --epsilon 0.05 --episodes 100

# Random-action baseline
python scripts/evaluate.py --baseline-only --episodes 100
```

We always evaluate at `--epsilon 0.05` rather than 0 because pure greedy play locks the agent into the same trajectory every episode, which makes per-episode completion rates degenerate. ε=0.05 matches the noise level the policy was trained against.

A quirk worth knowing: the `best.pt` file is saved the first time any episode achieves a new minimum y, which is often a one-off lucky episode rather than the trained policy's actual peak. For our results we evaluated specific late-stage checkpoints (`checkpoint_ep5000.pt` for dqn_r1, `checkpoint_ep2000.pt` for v3_r9). If you train your own runs, expect to do the same.

## Comparing models

To run all the comparisons in one batch and produce paper-ready plots:

```bash
python scripts/compare_all.py
```

This auto-discovers every model under `runs/` (both `runs/{id}/best.pt` and flat `runs/{id}.pt` layouts), evaluates each at ε=0.05 over 100 episodes, and writes:

- `docs/comparison/comparison_table.csv` — one row per method
- `docs/comparison/summary.json` — same data, JSON
- `docs/comparison/completion_bar.png` — bar chart of completion rates, sorted
- `docs/comparison/outcome_breakdown.png` — stacked bar showing complete / died / timeout per method
- `docs/comparison/height_distribution.png` — mean min-y reached per method

If you've manually picked specific checkpoints (rather than relying on `best.pt`), pass them as overrides:

```bash
python scripts/compare_all.py \
  --override dqn_r1=runs/dqn_r1/checkpoint_ep5000.pt \
  --override v3_r9=runs/v3_r9/checkpoint_ep2000.pt
```

## Watching and recording

```bash
./watch.sh -i dqn_r1                  # auto-loads runs/dqn_r1/best.pt
./watch.sh -m runs/dqn_r1/checkpoint_ep5000.pt -e 10
```

The watch script defaults to ε=0.10 — the trained policies need a bit of noise to get out of any deterministic stuck states they might lock into.

To record a successful run as a GIF (for slides, blog posts, etc.):

```bash
python scripts/record_gif.py --run-id dqn_r1
```

It retries until it captures a completing episode — you don't get a GIF of a dead agent.

## Reproducing our results

The full pipeline takes roughly 9–10 hours of GPU time end-to-end:

```bash
# 1. Regenerate TAS expert data through the current 87-dim env (~30 sec)
python scripts/export_tas.py --room 0

# 2. Train all five methods (each 2–3 hours on GPU)
python src/train.py            --run-id dqn_r1        --device cuda --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
python src/train_v3.py         --run-id v3_r9         --device cuda --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
python src/train_bc.py         --run-id bc_r2         --device cuda --epochs 200
python src/train_hybrid.py     --run-id hybrid_r2     --device cuda --episodes 5000 --epsilon-decay 0.999990 --epsilon-end 0.05
python src/train_curriculum.py --run-id curriculum_r5 --device cuda

# 3. Batch-evaluate everything
python scripts/compare_all.py
```

Or skip retraining by downloading our checkpoints from the [GitHub release](https://github.com/jmtorr3/celeste-rl/releases) and dropping them into `runs/`.

## What we found

This is a snapshot of the comparison table from `compare_all.py` (ε=0.05, 100 episodes per model):

| Method | Run | Architecture | Completion | Notes |
|---|---|---|---|---|
| **Plain DQN, semantic** | **dqn_r1** | DQN | **57%** | winner |
| Dueling DQN + curiosity, semantic | v3_r9 | Dueling | 37% | works but underperforms plain |
| Random baseline | — | — | 0% | floor |
| Behavioral Cloning | bc_r2 | DQN | 0% | mostly timeouts (deterministic loop) |
| Curriculum, 6 stages | curriculum_r5 | Dueling | 0% | doesn't generalize back to full level |
| Hybrid (BC + DQN) | hybrid_r2 | DQN | 0% | impl. bug — see Training section |
| DQN, **raw tile IDs** | v3_r8 | Dueling | 0% | the "before perception fix" baseline |

Single-seed numbers carry roughly ±10 points of variance — dqn_r1 measured 68% in an isolated 100-episode evaluation and 57% in a batched run on the same checkpoint. The qualitative ordering is stable.

The interesting part is *how* each method fails. BC and curriculum agents lock into deterministic loops and time out (88+ timeouts out of 100 episodes). Hybrid actively walks into spikes (100% deaths — worse than random) because of the zero-reward bug. Plain DQN's failures are more evenly split between early-game ε-noise deaths and "almost made it" timeouts at the upper ledge. The breakdown chart from `compare_all.py` shows this visually.

## A few things to know if you're picking this up

The `< -8` height threshold you might see in older code or in stale comments is wrong. The real exit fires at `player.y < -4` strictly. Our trainers use `info['completed']` (which calls into the env's room-transition detector); anything that gates on `max_height < -8` will report 0% completion even when the agent is actually finishing the room.

Old checkpoints from before the perception fix (state went from 31-dim to 87-dim) can't be loaded by the current code. If you find a checkpoint that errors out on a state-dim mismatch, that's why.

The `pyleste/` directory is a vendored copy of the [Pyleste emulator](https://github.com/CelesteClassic/Pyleste). We modified some of its internal state-access paths to expose what `CelesteEnv` needed; we did not modify game logic.

## References

- [Pyleste](https://github.com/CelesteClassic/Pyleste) — Python emulator we built on
- [Celeste Classic](https://celesteclassic.github.io/) — original game, web-playable
- [TAS database](https://celesteclassic.github.io/tasdatabase/classic/) — TAS source
- Mnih et al., *Playing Atari with Deep Reinforcement Learning* (DQN), [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)
- Bellemare et al., *Unifying Count-Based Exploration and Intrinsic Motivation*, [arXiv:1606.01868](https://arxiv.org/abs/1606.01868)
- [`effdotsh/Celeste-Bot`](https://github.com/effdotsh/Celeste-Bot) — independent GA-based Celeste agent that solves the full game; useful counterpoint to our DQN-family negative results
