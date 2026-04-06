# Methodology

## 1. Dataset

### Source

Our primary data source is a community-maintained Tool-Assisted Speedrun (TAS) database for Celeste Classic, accessed via the [UniversalClassicTas](https://github.com/CelesteClassic/UniversalClassicTas) repository. TAS recordings are frame-perfect playthroughs created by human experts using save-states and slow-motion tools to produce optimal or near-optimal play.

### Structure

Each TAS recording is stored as a sequence of integer bitfields, one per game frame (60 Hz). Each integer encodes the full button state for that frame using a bitmask over the PICO-8 input set: Left, Right, Up, Down, Jump, and Dash. A single playthrough of Celeste Classic spans roughly 2,500–5,000 frames depending on route and execution.

We parse each bitfield into a binary action vector and pair it with the corresponding game state extracted by stepping the Pyleste emulator forward. This produces a labeled dataset of `(state, action)` pairs suitable for supervised imitation learning (Behavioral Cloning).

### Preprocessing and Feature Extraction

Raw game state is extracted from the Pyleste emulator at each frame and converted into a normalized observation vector:

| Feature | Raw Value | Normalized |
|---|---|---|
| Player x | pixel coordinate [0, 128] | `x / 128.0` |
| Player y | pixel coordinate [0, 128] | `y / 128.0` |
| Horizontal velocity | float | `spd.x / max_spd` |
| Vertical velocity | float | `spd.y / max_spd` |
| Grace frames | integer [0, 6] | `grace / 6.0` |
| Dash availability | binary {0, 1} | unchanged |

> **Planned expansion (per professor feedback):** We will append a flattened 5×5 tile grid centered on the player, expanding the state from 6 to 31 dimensions. Each tile is encoded as `0.0` (empty), `1.0` (solid platform), or `-1.0` (hazard/spike).

Raw TAS action bitfields are decoded into one of 15 discrete action indices by filtering out physically meaningless button combinations from the full 64-combination space.

For Behavioral Cloning, Gaussian noise (`σ = 0.01–0.05`) will be injected into the four kinematic features (x, y, spd.x, spd.y) during training to reduce overfitting to the frame-perfect TAS trajectory. The discrete features (grace, djump) are not perturbed.

### Data Patterns and Relevance

TAS data is highly non-uniform: the agent spends the majority of frames moving right and upward, with dash actions concentrated at platform edges and gap crossings. This class imbalance means a naive BC model will rarely learn to dash. We account for this through action-weighted loss during BC training. Additionally, because TAS play is near-optimal, the state distribution it covers is narrow — the agent never enters the recovery states a suboptimal policy would need to handle, motivating the hybrid approach described below.

---

## 2. Models

### 2.1 Deep Q-Network (DQN)

**How it works.** DQN is a value-based RL algorithm that learns a function `Q(s, a)` estimating the expected discounted return from taking action `a` in state `s`. At each step the agent selects the action with the highest Q-value (with ε-greedy exploration) and stores the transition `(s, a, r, s')` in a replay buffer. A separate target network — a periodically updated copy of the main network — is used to compute stable Bellman targets:

```
y = r + γ · max_{a'} Q_target(s', a')
```

The main network is trained by minimizing the Huber loss between predicted Q-values and these targets via mini-batch gradient descent.

**Architecture.**

```
Input (6 or 31) → Dense(256, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Output (15 Q-values)
```

We also implement **DuelingDQN**, which splits the final layers into a state-value stream `V(s)` and an advantage stream `A(s, a)`, combined as `Q(s,a) = V(s) + A(s,a) − mean(A)`. This stabilizes training by decoupling the value of being in a state from the relative merit of each action.

**Mathematical foundations.** DQN is grounded in temporal-difference learning and the Bellman optimality equation. The key insight is that the optimal Q-function satisfies:

```
Q*(s, a) = E[r + γ · max_{a'} Q*(s', a')]
```

Experience replay breaks correlations between consecutive transitions; the target network reduces the moving-target problem that makes naive Q-learning diverge with neural network approximators.

**Hyperparameters.**

| Parameter | Value |
|---|---|
| Episodes | 3,000 |
| Max steps / episode | 500 |
| Learning rate | 0.0005 (Adam) |
| Discount factor γ | 0.99 |
| ε start → end | 1.0 → 0.05 (decay 0.9995) |
| Batch size | 128 |
| Replay buffer capacity | 200,000 |
| Target network update | every 200 steps |
| Gradient clip | max norm 10 |

**Reward shaping.** Celeste Classic provides no intermediate reward signal, so we define a shaped reward:

| Event | Reward |
|---|---|
| New maximum height reached | `+1.0 × height_gained` |
| Exploring a new grid position | `+0.5` |
| Stuck (≥ 100 steps without progress) | `−0.2` |
| Death | `−5.0` |
| Level complete | `+100.0` |

**Expected performance.** We expect the DQN agent to learn consistent upward movement and basic platform navigation within 3,000 episodes, but to struggle with the precise dash-jump sequences required for later screens. It serves as our primary RL baseline.

**Implementation challenges.** The main difficulty is reward sparsity: without height shaping the agent rarely discovers the level-complete signal. The exploration bonus and stuck penalty are essential to prevent the agent from settling into degenerate loops at the start position.

**Limitations.** DQN with a 6-dimensional state has no knowledge of nearby terrain, making it unable to anticipate hazards it cannot see in its kinematic state alone. The planned tile-grid expansion addresses this.

---

### 2.2 Behavioral Cloning (BC)

**How it works.** Behavioral Cloning treats imitation learning as supervised classification. The model is trained to map observed game states to the action taken by the TAS expert at that state. At inference time the policy selects actions greedily from the learned distribution, with no exploration or online interaction during training.

**Architecture.**

```
Input (6 or 31) → Dense(256, ReLU) → Dense(256, ReLU) → Output (15 logits) → Softmax
```

Trained with cross-entropy loss, weighted by inverse action frequency to address TAS class imbalance.

**Mathematical foundations.** BC minimizes the negative log-likelihood of the expert's actions under the learned policy:

```
L(θ) = −(1/N) Σ log π_θ(a_i | s_i)
```

This is equivalent to minimizing the KL divergence between the expert policy and the learned policy over the training state distribution.

**Expected performance.** BC is expected to reproduce the TAS trajectory closely on the sections of state space covered by the demonstrations, but to fail badly on states the TAS never visits (distribution shift / covariate shift problem). It serves as a performance ceiling for what pure imitation can achieve.

**Limitations.** Compounding errors: small deviations from the expert trajectory place the agent in out-of-distribution states where the BC policy has no reliable guidance. It also cannot improve beyond the TAS demonstrator.

---

### 2.3 Hybrid (DQN + TAS Pre-loading)

**How it works.** The hybrid agent initializes its DQN replay buffer with transitions extracted from TAS recordings before any online training begins. The DQN then continues training with its standard ε-greedy loop. This gives the agent a warm-start from expert demonstrations while still allowing it to learn from its own experience and generalize beyond the TAS trajectory.

**Mathematical foundations.** Equivalent to prioritized experience replay where expert transitions receive equal initial priority to online transitions. The agent optimizes the same Bellman objective as standard DQN; the difference is purely in the initial composition of the replay buffer.

**Expected performance.** We hypothesize the hybrid agent will outperform both DQN (better early guidance) and BC (can recover from distribution shift) — this is the central claim of the project.

**Challenges.** Catastrophic forgetting: as online transitions fill the buffer, expert transitions are eventually displaced. We will experiment with reserving a fixed fraction of the buffer for expert data.

---

## 3. Evaluation Metrics

### Metrics

| Metric | Description |
|---|---|
| **Success rate** | Fraction of episodes in which the agent completes the level |
| **Mean episode reward** | Average total shaped reward per episode (50-episode moving average) |
| **Best height reached** | Maximum vertical progress achieved across all episodes |
| **Mean height reached** | Average maximum height per episode |
| **Death rate** | Fraction of episodes ending in death (vs. timeout) |
| **Steps to completion** | Mean number of steps on successful episodes |

### Baselines

We compare all three learned agents against two reference baselines:

1. **Random policy** — selects uniformly at random from the 15 actions at each step.
2. **TAS reference** — the expert playthrough itself, providing an upper bound on performance.

### Desired Output

The ideal output is an agent that consistently completes room 0 of Celeste Classic (the first screen) within the step budget. Secondary success is reaching the top platform without completing the room.

### Expected Accuracy

- **Random baseline:** near-zero success rate, mean height ≈ 20–30 px.
- **DQN:** success rate 5–20%, mean height 60–90 px after 3,000 episodes.
- **BC:** success rate 30–60% on in-distribution states, degrades quickly off-trajectory.
- **Hybrid:** success rate 25–50%, more robust than BC off-trajectory.
- **TAS:** 100% success rate by definition.

Success rate and mean height are the primary metrics for the final comparison. Because all metrics are computed by running agents in the deterministic Pyleste emulator, there is no subjectivity in measurement — the same random seed produces identical results across runs.
