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

The agent observes a 6-dimensional state:
- Player position (x, y) — normalized
- Player velocity (spd.x, spd.y) — normalized
- Grace frames (coyote time)
- Dash availability

And can take 15 actions:
- Nothing, Left, Right
- Jump, Jump+Left, Jump+Right
- Dash in 8 directions

### Reward Function

```python
reward = 0
if reached_new_height:
    reward += height_gained * 1.0  # Bonus for upward progress
if is_moving:
    reward += 0.01                  # Small exploration bonus
if stuck_too_long:
    reward -= 0.1                   # Penalty for inaction
reward -= 0.01                      # Time penalty per step
# Death: -5.0 | Level complete: +100.0
```

### Network Architecture (`src/network.py`)

Two architectures available:
- **DQN**: `Input(6) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Output(15)`
- **DuelingDQN**: Shared features → separate value and advantage streams

## Quick Start

### Train an agent
```bash
./train.sh                              # DQN, 3000 episodes, room 0
./train.sh -v2                          # train_v2 (exploration bonuses)
./train.sh -e 5000 -r 2                 # 5000 episodes, room 2
./train.sh -m models/dqn_best.pt        # continue from checkpoint
./train.sh --eval-only -m models/dqn_best.pt
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
