# celeste-rl

Deep reinforcement learning agent that learns to play Celeste Classic. Uses DQN with custom reward shaping built on the Pyleste emulator.

## Project Structure

```
celeste-rl/
├── README.md
├── requirements.txt
├── train.sh                    # Train agent (shortcut)
├── watch.sh                    # Watch agent play (shortcut)
│
├── pyleste/                    # Pyleste emulator
│   ├── PICO8.py
│   ├── CelesteUtils.py
│   ├── Searcheline.py
│   └── Carts/
│       └── Celeste.py
│
├── src/                        # RL code
│   ├── __init__.py
│   ├── environment.py          # CelesteEnv class (Gymnasium-style wrapper)
│   └── network.py              # DQN and DuelingDQN architectures
│
├── scripts/                    # Utility scripts
│   ├── watch_agent.py          # Visualize trained agent
│   └── evaluate.py             # Evaluate performance
│
├── configs/
│   └── dqn_config.yaml         # Hyperparameter config
│
├── models/                     # Saved model weights (gitignored)
│   └── .gitkeep
│
├── notebooks/                  # Jupyter notebooks
├── tests/
└── docs/
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jmtorr3/celeste-rl.git
cd celeste-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How It Works

### Environment (`src/environment.py`)

The agent observes a **31-dimensional state**:
- Player position (x/128, y/128) — normalized to [0, 1]
- Player velocity (spd.x/4, spd.y/4)
- Grace frames (coyote time, /6)
- Dash availability
- 5×5 tile grid centered on player (25 values from `game.tile_at`)

And can take 15 actions:
- Nothing, Left, Right
- Jump, Jump+Left, Jump+Right
- Dash in 8 directions

### Reward Function

```python
# Progressive height bonus — stronger near the exit
progress_scale = max(1.0, (96 - player.y) / 24.0)  # 1x at start, 4x near exit
reward += height_gained * progress_scale

# Milestone bonuses (first time per episode)
# y<40: +20 | y<20: +40 | y<10: +80 | y<0: +150 | y<-5: +300

# Small bonuses/penalties
reward += 0.01   # movement bonus
reward -= 0.1    # stuck penalty (after 30 stationary frames)
reward -= 0.01   # time penalty per step

# Terminal rewards
# Death: -5.0 | Level complete: +500.0

# Early termination if stuck_count > 150
```

### Network Architecture (`src/network.py`)

Two architectures available (train_v3 uses DuelingDQN + Double DQN):
- **DQN**: `Input(31) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Output(15)`
- **DuelingDQN**: Shared features → separate value and advantage streams, combined as `Q = V + (A - mean(A))`

## Quick Start

### Train an agent
```bash
# Always pass a unique --run-id to avoid overwriting models from other runs
python src/train_v3.py --episodes 5000 --epsilon-decay 0.999990 --device cuda --run-id my_run

# Resume from checkpoint
python src/train_v3.py --resume models/my_run_checkpoint_ep500.pt --run-id my_run --device cuda

# Eval only
python src/train_v3.py --eval-only --model models/my_run_best.pt
```

### Watch a trained agent play
```bash
./watch.sh                              # loads models/dqn_best.pt
./watch.sh -m models/dqn_final.pt      # specific model
./watch.sh -e 10 -d 0.05               # 10 episodes, slower playback
./watch.sh -r 1                         # different room
```

## Scripts

### Manual invocation
```bash
python scripts/watch_agent.py --model models/dqn_best.pt --room 0 --delay 0.03
```

### Evaluate performance
```bash
python scripts/evaluate.py --model models/dqn_best.pt --episodes 100 --baseline
```

## Configuration

Key hyperparameters in `configs/dqn_config.yaml`:

```yaml
environment:
  room: 0
  max_steps: 500

agent:
  learning_rate: 0.0005
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.9995
  batch_size: 128
  buffer_size: 200000
  target_update_freq: 200
```

## References

- [Pyleste](https://github.com/CelesteClassic/Pyleste) - Python Celeste Classic emulator
- [Celeste Classic](https://celesteclassic.github.io/) - Original game
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Playing Atari with Deep RL
