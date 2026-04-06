# Professor Feedback — Required Fixes

## 1. Relative Coordinates

**Problem:** `_get_obs()` uses `player.x / 64 - 1` and `player.y / 64 - 1`. While Celeste Classic rooms are already local (0–127 px), the normalization formula is non-standard and wasn't communicated clearly as screen-relative.

**Fix:** Change to `player.x / 128.0` and `player.y / 128.0` (clean [0, 1] normalization within the 128×128 px room). Alternatively, use spawn-relative coords `(player.x - spawn_x) / 128.0` for stronger generalization across rooms.

**Files:**
- `src/environment.py` — `_get_obs()` lines 164–165
- `src/train_v2.py` — `CelesteEnvV2._get_obs()` (own hardcoded obs)
- `scripts/export_tas.py` — inline obs block in `record_attempt()` lines 75–86

---

## 2. Data Augmentation (Noise) for Behavioral Cloning

**Problem:** TAS data is frame-perfect, so a BC model trained on it may overfit exact state trajectories and fail when even tiny deviations occur at runtime.

**Fix:** During BC training, inject Gaussian noise into the kinematic state features before computing the loss:

```python
noise_std = 0.02
state_noisy = state.clone()
state_noisy[:, :4] += torch.randn_like(state[:, :4]) * noise_std  # x, y, spd_x, spd_y
```

Suggested noise levels: σ = 0.01–0.05 on `x_rel, y_rel, spd_x, spd_y`. Do NOT add noise to `grace` or `djump` (discrete game-logic values).

**Files:**
- BC training script (not yet written) — apply noise in the training loop before `model(state)`

---

## 3. Surrounding Environment Grid (Tile Observations)

**Problem:** The state `(x, y, spd_x, spd_y, grace, djump)` has no information about nearby terrain. The agent is effectively blind to walls, spikes, and platforms.

**Fix:** Append a flattened N×N tile grid centered on the player to the state vector. Recommended: 5×5 (radius = 2 tiles).

**Tile encoding:**
- `0.0` — empty
- `1.0` — solid (tile flag 0 in PICO-8)
- `-1.0` — hazard (spike tiles: 17, 27, 43, 59)

**Implementation in `_get_obs()`:**

```python
r = self.grid_radius  # default 2 → 5x5 grid
tx0 = int(player.x) // 8
ty0 = int(player.y) // 8
game = self.p8.game
for dy in range(-r, r + 1):
    for dx in range(-r, r + 1):
        tx = max(0, min(15, tx0 + dx))
        ty = max(0, min(15, ty0 + dy))
        tile = game.tile_at(tx, ty)
        if self.p8.fget(tile, 0):       # solid
            obs.append(1.0)
        elif tile in (17, 27, 43, 59):  # spikes
            obs.append(-1.0)
        else:
            obs.append(0.0)
```

**State dimension change:** 6 → 31 (6 kinematic + 25 tile grid)

**Files:**
- `src/environment.py` — `_get_obs()`, `_get_obs_dim()`, add `grid_radius` param to `__init__`
- `src/train_v2.py` — same changes for `CelesteEnvV2`
- `scripts/export_tas.py` — update inline obs + `DQNAgent(state_dim=31, ...)`
- `src/agent.py` — update `state_dim=6` → `31` in `__main__` test block
- `src/network.py` — update `state_dim=6` → `31` in `__main__` test block

---

## Proposal Text Updates (`Project_Proposal.tex`)

In **Section 2**, update the state vector description:

> `s_t = (x_{rel}, y_{rel}, \text{spd}_x, \text{spd}_y, \text{grace}, \text{dash}, \mathbf{g})` where `x_{rel}, y_{rel} \in [0,1]` are screen-relative pixel coordinates and `\mathbf{g} \in \{-1, 0, 1\}^{25}` is a flattened 5×5 grid of local tile types (empty, solid, hazard), yielding a 31-dimensional state vector.

Also add a sentence about data augmentation:

> To mitigate overfitting to the frame-perfect precision of TAS demonstrations, Gaussian noise ($\sigma \approx 0.02$) is injected into the kinematic features during behavioral cloning training.
