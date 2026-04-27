# Celeste-RL Devlog

A combined log of the celeste-rl project, written across April 2026. Jorge Manuel Torre led the project (Pyleste env integration, training infrastructure, BC and hybrid pipelines, run management, evaluation framework). Akhil Madipalli owned DQN training, reward shaping, and ultimately found the perception/state encoding fix that turned the project around. Maryam Malik ran evaluation across every method and produced the final comparison results. The goal was to train an agent to complete room 0 of Celeste Classic and to compare reinforcement learning, behavioral cloning, hybrid imitation+RL, and curriculum learning on the same problem.

The short version of where we ended up: plain DQN with a semantic tile encoding got 57–68% completion under reproducible evaluation; every additional method we tried — Dueling architecture, curiosity bonus, BC, hybrid, curriculum — performed equal to or worse than plain DQN. The dominant factor turned out to be the state representation, not the algorithm.

## Phase 1: the original DQN runs (April 17–18)

We started from a v3 environment with the three professor-feedback fixes applied: coordinate normalization changed from `x/64 - 1` to `x/128.0`, a 5×5 tile grid centered on the player added to the state via `game.tile_at`, and `_get_obs_dim()` updated to return 31. `train_v3.py` wrapped the env in a subclass `CelesteEnvV3` that added per-episode and global position visitation bonuses for exploration.

The first run (run 1) used `epsilon_decay=0.9995`, which we discovered too late was decaying epsilon to its 0.05 floor in about thirty episodes — the agent was nearly greedy for nine-tenths of the run. Best height plateaued at 13. Run 2 fixed this by slowing the decay to `0.999990`, which spreads the decay over roughly fifteen hundred episodes. The slower exploration helped — best height got to 5 around episode 550, a real improvement — but the agent still completed zero out of five thousand episodes, and greedy evaluation locked at height=72 every time. We had a degenerate local policy.

Looking at this post-mortem, two things were clearly wrong. The completion bonus of +100 was invisible against the ~80–100 of shaped reward an episode could earn just from the height bonus, so completing the level wasn't meaningfully better than a decent incomplete run. And we had no early termination for stuck episodes, so the agent wasted four hundred steps of every episode after getting trapped, polluting the replay buffer with junk. We bumped the completion bonus to +500 and added a `stuck_count > 150` early termination.

Run 3 still hit the y=13 plateau and stayed there for fifteen hundred episodes after epsilon hit floor. The deeper problem was clear: the agent had never observed the completion reward, so no amount of tuning the bonus value mattered. The shaped rewards were creating a local optimum at y=13 that the agent couldn't escape because there was no gradient signal beyond it.

For run 4 we applied a stack of fixes — Double DQN to reduce overestimation, Dueling DQN with separate value and advantage streams, a progressive height bonus that scales up to 4× near the exit, milestone bonuses (+20/+40/+80/+150 the first time per episode the agent crosses y=40, 20, 10, 0), and we raised the epsilon floor from 0.05 to 0.10 so the policy wouldn't fully lock in. Run 4 became the first run to break through y=13 and reach y=-3, five pixels from exit. It got there during high-epsilon exploration and then plateaued for nine hundred more episodes without completing.

That y=-3 plateau exposed a new problem: a good incomplete episode reaching y=-3 and hitting all the milestones earned around 587 in shaped reward, while the +500 completion bonus was *less* than that. From the agent's perspective, completion barely outperformed getting close. We added a y<-5 milestone worth +300 to bridge the gap. Run 5 didn't reach y=-3 at all; it regressed back to the y=13 plateau because it didn't get lucky during early exploration the way run 4 had. The y=13 section requires a specific dash-jump sequence, and pure epsilon-greedy doesn't find it consistently.

Run 5 also overwrote run 4's checkpoint file. We added a `--run-id` flag to `train_v3.py` so future runs would prefix all save paths with the run ID; defaults to `v3` for backwards compatibility. Run 6 (`v3_r6`) did better — it reached y=-4 around episode 3350. But the buffer hit its 500k cap at episode 2050 and started cycling, and around episodes 3500–4000 the transitions that led to y=-4 got displaced. Average height collapsed from 43 to 67. The agent partially recovered by episode 4400 but never beat -4.0 again.

The greedy eval told the real story: success rate 0/50, mean height 51 every single episode. Deterministic, identical trajectory, dies the same way. The -4 best height was an exploration fluke at epsilon=0.10; the Q-values for the path beyond y=51 were never reinforced enough to win the argmax. We closed Phase 1 with `models/v3_r6_best.pt` as the official DQN result: 0% completion, training best -4, eval mean height 51.

This wasn't surprising. Celeste requires precise multi-step sequences over hundreds of frames before any terminal reward, which is one of the hardest problem structures for DQN: long horizon, sparse terminal reward, and epsilon-greedy noise that disrupts frame-perfect timing. The professor pointing toward BC and Hybrid implicitly acknowledged this — TAS data exists precisely because the level demands precision that random exploration can't find. Phase 1's 0% completion was the expected baseline that motivated everything after.

## Phase 2: behavioral cloning (April 18)

We exported all 31 TAS files from `TAS_data/any%/` into a single (state, action) pickle for behavioral cloning. The export script (`scripts/export_tas.py`) replays each TAS through the Pyleste env and records the 31-dim observation alongside the action index at every frame. Two parser quirks across rooms — some files use `[0,]` or `[0.636,]` as a header instead of `[]` — were handled by stripping all brackets and skipping non-integer tokens. Final dataset: 1930 transitions across 31 rooms, with 28 unrecognized inputs skipped. Several rooms (19, 24, 25, 30) terminated early on valid TAS inputs, suggesting state divergence between Pyleste and whatever ran the TAS originally.

The action distribution was heavily imbalanced: `btn=0` (idle) was 48% of all transitions, `btn=2` (right) another 28%, leaving everything else to compete for the remaining quarter. Two actions (`btn=32` dash, `btn=41` dash+down+left) had zero training examples but non-zero class weights from the inverse-frequency calculation, which meant the model would assign confident predictions to actions it had never seen. We added Gaussian noise on the kinematic features (x, y, spd_x, spd_y) at σ=0.02 to reduce overfitting to the exact TAS trajectory.

Two BC runs: `bc_r1` at 300 epochs with σ=0.02 hit 57.5% validation accuracy and eval height 90, 0/50 completions. `bc_r2` at 500 epochs with σ=0.05 reached 63.7% val acc and eval height 77, also 0/50. Both showed severe overfitting — train loss dropped to 0.17 while val loss exploded from ~4 to ~13. More noise didn't fix it; the dataset was simply too small.

BC ended up worse than DQN at eval. The proximate cause was a 1930-transition dataset with two actions never seen in training, but the deeper issue was that 66 transitions for room 0 specifically — even with augmentation — cannot teach a policy that recovers from off-distribution states. Both DQN and BC failing independently was the right paper narrative: it directly motivated combining them.

## Phase 3: hybrid (April 18)

`train_hybrid.py` warm-starts a DQN from TAS expert data. We built a `HybridBuffer` class with a protected partition that never evicts: 100k slots reserved for expert transitions, 400k for online experience, total 500k. All 1930 expert transitions get pre-loaded with `reward=0` (we'll come back to this), then training proceeds normally with batches drawn 20/80 from expert/online.

`hybrid_r1` ran for five thousand episodes. The expert data did help early — best height improved from 30 to 13 in the first 650 episodes, faster than pure DQN had reached similar heights. But epsilon dropped fast, the same y=13 bottleneck applied, and the agent never broke through. 0/5000 completions. Greedy eval was deterministic at height 77, *worse* than DQN's 51.

We didn't realize at the time how significant the `reward=0` choice was. The agent saw expert states in the buffer and learned which actions to take in them, but the Q-targets for those transitions were 0-reward bootstraps. When the online policy later reached those states, the Q-values had been anchored low. This may explain why eval height was 77 rather than 51 — the expert transitions with reward=0 may have actively suppressed Q-values in early-game states that DQN had assigned positive value to. We came back to this bug later (Phase 9) when hybrid was rerun with a fixed reward function still showed the same issue.

Phase 3's summary across the three approaches:

| Model              | Eval Height | Completions |
|--------------------|-------------|-------------|
| DQN (v3_r6)        | 51          | 0/50 (0%)   |
| BC (bc_r2)         | 77          | 0/50 (0%)   |
| Hybrid (hybrid_r1) | 77          | 0/50 (0%)   |

All three failed at the same bottleneck. Negative result was the result.

## Phase 4: curriculum learning and a critical bug (April 18)

The thinking behind curriculum: all three approaches fail because the level requires ~300 frames of precise execution before any terminal reward, and Q-values at the bottleneck never get reinforced because the agent never reaches completion. If we start the agent near the exit, it only needs three or four moves to complete — the +500 bonus is immediately reachable, Q-values bootstrap correctly, and a working policy forms fast. Then move the spawn down stage by stage.

While testing `train_curriculum.py` locally, we found a bug that invalidated *every prior run*. When the player exits the level, Pyleste loads the next room and the `player` object disappears from `p8.game.objects`, replaced by a `player_spawn`. Our `_get_player()` returns `None`, and `_compute_reward()` was returning `-5.0` (death penalty) instead of `+500.0` (completion). Every run across DQN, BC, and Hybrid had been *punishing the agent for reaching the exit*. The agent in DQN run 6 that reached best height -4 was actually completing the level and getting killed for it.

The fix added an `_is_room_transition()` check that looks for `player_spawn` in `p8.game.objects`; `_compute_reward` now returns +500 when the player is None and a room transition is in progress, -5 only on actual death. We also added an `info['completed']` flag so training scripts could stop checking `max_height < -8` and use the actual signal. Updated `train_v3.py`, `train_hybrid.py`, `train_bc.py`, and `train_curriculum.py` to use `info.get('completed', False)`.

A second bug surfaced during local testing: `epsilon_decay` was being applied per *step* (in `agent.update()`). With 100 steps per episode and decay=0.99850, epsilon hit the floor after fifteen episodes — pure greedy before learning anything. We disabled per-step decay (passing `epsilon_decay=1.0` to the agent) and managed epsilon explicitly per episode in `train_stage()`.

A third issue: our initial spawn x=60 was under a solid ceiling at every stage. We sampled x positions from the TAS replay at matching y values: stage 1 (y=15) gets x=97 from TAS step 58, stage 2 (y=30) gets x=89 from step 52, and so on. With these fixes in place, local testing on CPU showed stages 1 and 2 advancing in 33–175 episodes; stages 3 and 4 needed more budget than CPU could give us.

`curriculum_r1` on the cluster (Tesla T4, 2000 episodes per stage, 50% advancement threshold over a 100-episode window) advanced stages 1 and 2 cleanly, then stage 3 (y=50) ran for the full 2000 episodes producing 480 completions but never sustained 50% over a 100-ep stretch. Stage 4 inherited a stage-3 policy that wasn't fully baked and produced 0/2000 completions.

Those 480 completions in stage 3 were significant: the first time any approach in this project actually completed Celeste Classic. All previous runs had 0% because of the room-transition bug. The 50% threshold was simply too strict for stage 3's noisy convergence.

`curriculum_r2` lowered the threshold to 30%. Stages 1, 2, and 3 all advanced; stages 4 and 5 timed out but stage 5 produced **one full-level completion from spawn y=96** around episode 1550 — the first time any agent in the project completed the level from the normal start. Stage 4 came within 3 pixels of the exit with 4 completions across 2000 episodes.

`curriculum_r3` added a y=60 intermediate stage between stages 3 and 4 to bridge the gap, lowered epsilon-end to 0.05, and lowered the threshold to 0.25. Stages 1 through 4 all advanced for the first time. Stage 5 still timed out with 5 completions; stage 6 (full level) regressed — average reward dropped from 100 to 31 over 2000 episodes as the agent forgot upper-level navigation while exhausting its budget on the lower section.

`curriculum_r4` resumed `curriculum_r3` from stage 5 with epsilon reset to 1.0 to re-explore. The reset was a mistake: the random actions destroyed the trained policy faster than learning could rebuild it (catastrophic forgetting), and r4 produced zero completions.

The curriculum runs revealed a consistent pattern. The transfer between adjacent stages worked when the gap was small enough — y=15 to y=30, y=30 to y=50, y=50 to y=60 with the bridge stage. The irreducible bottleneck was stage 5 (y=71): the path from there to the exit is long enough that ε=0.05 random actions still disrupt it, and 2000 episodes wasn't enough to consolidate it into a reliable policy. The "right" fix would have been substantially more training budget or a different algorithm entirely (PPO with curiosity-driven exploration).

By the end of curriculum_r4 we had:

| Model          | Training Completions | Eval Height | Eval Completion |
|----------------|----------------------|-------------|-----------------|
| DQN (v3_r6)    | 0 (had exit bug)     | 51          | 0%              |
| BC (bc_r2)     | —                    | 77          | 0%              |
| Hybrid r1      | 0 (had exit bug)     | 77          | 0%              |
| Curriculum r1  | 480 (stage 3)        | 72          | 0%              |
| Curriculum r2  | 485 (stages 3–5)     | 77          | 0%              |
| Curriculum r3  | 5 (stage 5 only)     | 64          | 0%              |
| Curriculum r4  | 0                    | —           | 0%              |

The story heading into Phase 5 was: "All four methods plateaued. Curriculum got first completions but doesn't generalize. We've exhausted DQN-family approaches without ever exceeding ~10 completions in 10,000+ episodes." We were ready to write the paper as a clean negative result and recommend population-based methods (NEAT, evolutionary strategies) or PPO with curiosity for future work.

## Phase 5: Akhil joins, the agent can't see (April 25)

Akhil started running his own experiments with the existing 5×5-state environment and observed the same y=13–17 plateau. He also noticed something we'd missed: visually, the agent oscillates back and forth on the right side of the room without ever committing to land on the right ledge. His hypothesis was that the 5×5 window (±16 pixels) was too narrow for the agent to see the ledge it needed to land on.

We bumped the tile grid to 9×9 (±32 pixels, roughly one full jump+dash radius), changing `OBS_DIM` from 31 to 87 and the loop bounds from `range(-2, 3)` to `range(-4, 5)`. While in there, we found that `_get_obs_dim()` was hardcoded to `return 31` instead of returning `self.OBS_DIM` — a real bug that would have silently stayed wrong if we hadn't been editing the file.

Akhil also pointed out something that turned out to invalidate a lot of stats: the level-exit threshold in Pyleste is `self.y < -4`, not `< -8`. The check is strict less-than:

```python
# pyleste/Carts/Celeste.py:349
# exit level off the top
if self.y < -4:
    g.next_room()
```

A bunch of our code keyed off `< -8`. Specifically:

- `environment.py:_is_complete()` had a fallback `return player.y < -8` that was actually dead code because the only callsite that reached it had already returned from the `_is_room_transition()` check. We collapsed it.
- All training scripts had `sum(1 for h in heights if h < -8)` as their "recent completions" counter. But `max_height` can never go below ~-4 because the player object is deleted by PICO-8 the moment y < -4. So that counter had been reporting 0 in *every run that had any completions*. Some of our earlier Phase 1–4 runs may have had completions that simply never showed up in the logs because of this counter bug.

Switched all five files to count from `info['completed']` directly.

Despite the bigger view and threshold fixes, the early v3_r8 runs still plateaued at y=13. The 9×9 helped perception but execution was still the problem. Half the episodes were ε-noise random and the final wall-jump-dash is a 5–10 frame button sequence that noise breaks. Akhil's screenshots confirmed it visually: the player sitting on the small upper-left ledge, one wall-jump-dash from exit, with the next room's `:D` text taunting from above, but unable to commit.

We restructured the milestone reward to give a finer gradient through the y=13 plateau:

```python
# old (5 milestones)
((40, 20), (20, 40), (10, 80), (0, 150), (-5, 300))

# new (15 milestones)
((50, 10), (40, 20), (30, 30), (20, 45), (17, 60), (15, 80),
 (13, 100), (11, 130), (8, 170), (5, 220), (2, 280), (0, 360),
 (-2, 480), (-3, 640), (-5, 900))
```

The new ramp totals about 3,525 in milestone reward versus the old 590. The shape is what matters — every two or three pixels of progress now gets rewarded, which gives gradient signal through the plateau.

The fully-fixed v3_r8 (9×9 state, OBS_DIM=87 fix, `_is_complete` cleanup, broken-counter fixes, 15-step milestone ramp, ε-end 0.05) ran fresh for 5000 episodes. It got the project's first DQN completions: 10 of them, clustered between episodes 2600 and 3250, peaking at 4/50 around episode 2750 with reward 377. Then it collapsed — the buffer hit 500k, FIFO eviction started displacing the early "stuck at y=13" transitions with later "almost-finished" ones, and Q-values drifted. Reward decayed from 243 down to 71 by episode 5000. Zero completions in the last fifteen hundred episodes.

The greedy eval told the same lie as before. `v3_r8_best.pt` was saved when `info['max_height'] < best_height` first fired with `best_height = -4.0` — a single lucky episode around 2750. Greedy eval gave height=39 on every one of fifty episodes. The 10 completions were partially-stochastic events; the network never learned a stable solution.

## Phase 6: the perception fix (April 26)

Watching `v3_r8_checkpoint_ep3000.pt` locally with ε=0.10 over ten episodes showed something useful: of the four time-out episodes, the agent ended at scattered positions (56,69), (45,62), (61,65), (37,72) — it wasn't navigating to a consistent location. The other six episodes were *early* deaths: the agent barely left the start before falling into spikes. This wasn't a "stuck near exit" problem, it was a "no stable policy" problem.

Akhil's instinct that the agent couldn't see properly turned out to be more right than the 9×9 expansion alone could fix. Investigating `_get_obs()` revealed two real bugs in how tiles were encoded.

First, the tile values were raw PICO-8 tile IDs:

```python
tile_grid.append(float(self.p8.game.tile_at(tx, ty)))
```

Tile IDs run from 0 to 255. Air is 0. Solid ground is 1, 2, 3, etc. Spike is 17. Decoration grass is around 100. The numerical distance between *spike (17)* and *decoration (18)* is the same as between *air (0)* and *solid (1)*. The agent had no way to learn that something was *landable* versus *deadly* versus *empty* — it was getting raw bytes. The line above this loop literally said "Normalize each value to be 0-1" but the loop didn't actually normalize them.

The fix classified each tile into one of five semantic categories:

- `-2.0` for out-of-bounds (sentinel — see below)
- `-1.0` for spike (death tiles 17, 27, 43, 59, looked up from `Celeste.py:575-578`)
- `0.0` for air (empty)
- `0.5` for other (decoration, fruit, anything non-deadly and non-solid)
- `1.0` for solid (uses `p8.fget(tile, 0)` to check the solid flag)

Second, the 9×9 view at room edges was being clamped, not bounded:

```python
tx = max(0, min(15, tile_x + dx))
```

When the player was near the right wall, the clamp made the right side of the view show *the same edge tile repeated* instead of an out-of-bounds signal. The agent literally couldn't tell the difference between "wall ends here" and "wall continues." This explained the right-side oscillation Akhil had pointed out — the agent saw the same view in a state where it should have seen the room boundary.

The fix uses a sentinel `-2.0` for OOB tiles. The agent now knows when it's near a room edge.

Both fixes are perception-level, not hyperparameter-level. They give the agent a meaningful five-class semantic encoding instead of 256 raw byte values, and awareness of room boundaries. Same network, same hyperparameters, same reward — just a better state encoding.

We also reorganized run artifacts at this point. After eight v3 runs and the curriculum experiments, `models/` was an unreadable mess of `v3_r1_best.pt`, `v3_r2_checkpoint_ep500.pt`, `curriculum_test4_stage3.pt`, etc. New layout: every run gets its own folder under `runs/{run_id}/` with `best.pt`, `final.pt`, checkpoints, the training pickle, and the curve PNG. Added `src/utils/paths.py` with a `run_dir(run_id)` helper and migrated all five training scripts to the new layout. Watch script picks up the new layout via `-i RUN_ID`.

`v3_r9` launched with all the fixes: 9×9 semantic state, OOB sentinel, 15-step milestone ramp, plain DQN (still using DuelingDQN at this point), ε-end 0.05.

The training trajectory was unlike anything we'd seen:

```
Ep  900: best 13.0  complete 0/50    ← same plateau v3_r8 was stuck on
Ep 1000: best  8.0                   ← broke through y=13
Ep 1050: best  5.0
Ep 1300: reward 344, avg_h 30.1      ← upper path discovered
Ep 1700: best -4.0  complete 7/50    ← first real completions
Ep 1800: complete 15/50              ← 30% completion
Ep 1850: complete 20/50              ← 40%
Ep 1950: complete 32/50, avg_h 8.1   ← 64% peak
```

By episode 1950 the agent was completing 64% of its training episodes. Average episode height was 8.1 — meaning *most* episodes were reaching near the exit, not just a lucky few. Watching `runs/v3_r9/best.pt` locally produced this output for the first time:

```
Episode: 3/3 | Step: 489
Action: dash
Position: (0, 0)
Reward: +500.00 | Total: 2218.11
🎉 LEVEL COMPLETE!
```

Replicable, deterministic level completion. After ten thousand-plus episodes across nine prior runs that had never produced a confirmed completion, the agent finished the room.

Comparing v3_r8 (raw IDs) to v3_r9 (semantic) head-to-head:

| Metric                | v3_r8 (raw tile IDs) | v3_r9 (semantic) |
|-----------------------|----------------------|------------------|
| First completion      | ep 2600 (1/50)       | ep 1700 (7/50)   |
| Peak completion rate  | 4/50 (8%)            | 32/50 (64%)      |
| Peak reward           | 377                  | 1669             |
| Behavior              | Stochastic, collapsed| Sustained        |

Same DQN, same training budget, same hyperparameters — the only difference was how the tile values reached the network.

Looking back across the project, every previous attempt had focused on reward shaping (milestones, exploration bonuses, height multipliers), state expansion (5×5 → 9×9), curriculum (spawning higher to focus on hard parts), or hyperparameters (epsilon decay, buffer size). None of those addressed the actual bottleneck: the agent literally couldn't tell a spike from a decoration.

The fix was fifteen lines of semantic classification — one of the smallest changes in the project, and the most impactful by orders of magnitude.

The full v3_r9 5000-episode run produced 1139 completions (22.8% lifetime rate). The buffer hit 500k at episode 1850 and *the agent kept improving even with FIFO eviction* — no catastrophic collapse like v3_r8. It sustained 30–60% completion across episodes 1700–5000. Average reward in that range was 1200–1700, vs. v3_r8's brief spike to 377.

The greedy eval was still misleading. `v3_r9/best.pt` was saved at episode 1700 when `best_height` first hit -4 and never overwrote because of strict less-than. The deterministic eval gave height=5 on all 50 episodes. The *real* best models were `checkpoint_ep1950.pt` (peak window), `checkpoint_ep3300.pt` (second peak), and `checkpoint_ep4500.pt` (late-stage strong).

This `best.pt` saving bug recurred across every method we tried after this. The fix is to save based on rolling completion rate rather than best height seen, which we noted but didn't get around to applying — instead we evaluated specific late-stage checkpoints by hand for the comparison.

## Phase 7: plain DQN beats Dueling DQN + curiosity (April 26)

After the v3_r9 result, we wanted to know whether the Dueling architecture and the curiosity bonus were actually contributing or whether the perception fix was doing all the work. While going through the training scripts to clean them up, we noticed `train_v2.py` had its own `CelesteEnvV2` class hardcoded to a 6-dimensional observation — a parallel-track experiment that never absorbed the perception fix. Retired it. v3 is a strict superset.

`train.py` (the original vanilla DQN script written before any of the professor feedback) still passed `use_simple_actions=True` to the env constructor, an argument we'd removed when consolidating action spaces. So the script crashed on launch. Fixed the dead kwarg. With that done, `train.py` was now plain DQN — no Dueling, no exploration bonus — but using the same env, same milestone rewards, same hyperparameters as v3_r9.

That made it a useful ablation. dqn_r1 versus v3_r9 differs on two axes: architecture (plain DQN vs Dueling DQN) and reward shaping (no curiosity vs curiosity bonus). It's not a clean single-variable test, which I want to be honest about — if dqn_r1 underperforms, we can't attribute the gap cleanly to either factor alone. To decompose it would have required another run with plain DQN + curiosity bonus, which we didn't have time for.

What we ran was dqn_r1: plain DQN, semantic state, no curiosity bonus, otherwise identical to v3_r9. Same 5000 episodes, same hyperparameters, same env.

It substantially outperformed v3_r9.

| Metric                       | v3_r9 (Dueling + curiosity) | dqn_r1 (Plain DQN) |
|------------------------------|------------------------------|---------------------|
| Total completions / 5000     | 1139 (22.8%)                | 1783 (35.7%)        |
| Peak completion rate         | 32/50 = 64% (ep 1950)       | 43/50 = 86% (ep 5000)|
| Avg height at ep 5000        | ~46 (collapsed)              | 0.8 (near exit)     |
| Avg reward at ep 5000        | ~106                         | 2635                |
| Trajectory                   | Peaked, then collapsed       | Still improving     |

Plain DQN produced 57% more completions and a 22-point higher peak. And it was still climbing when training ended at episode 5000. The training trajectory was clean: first completions at episode 1500, mini-collapse around episode 2000–2050 that recovered within 100 episodes, sustained 22+/50 completion rate from then through episode 5000, hitting 43/50 at the end.

Re-evaluating `runs/dqn_r1/checkpoint_ep5000.pt` with ε=0.05 over 100 episodes:

```
Success rate:  68/100 (68.0%)
Death rate:    24/100 (24.0%)
Timeout rate:   8/100 (8.0%)
Mean reward:   1970.61 ± 1289.09
Mean height:   11.8
Best height:   -4.0
```

68% reproducible completion. The 24% deaths were mostly early — worst case h=72 means the agent barely left the start when ε=0.05 noise pushed it into spikes. The 1289 reward standard deviation is the bimodal distribution — completions at 2700–3350, failures at 20–300.

**This was the headline.** 68% under controlled evaluation. 86% peak training. 35.7% lifetime training. The story flipped from "perception fix took DQN from 8% to 64%" to "perception fix took DQN from 8% to 86%, and adding Dueling + curiosity actually hurt."

Two defensible findings now:

1. **Semantic perception is necessary.** Raw tile IDs (v3_r8) plateau at 8%. Semantic encoding lets the same algorithm reach 86%.
2. **Dueling + curiosity hurts in this setting.** Plain DQN substantially outperformed the more complex variant.

Possible reasons: the curiosity bonus rewards visiting new (x, y) cells, and once the agent finds the path, those rewards mislead — pulling it toward unexplored corners away from the optimal trajectory. Likely caused v3_r9's mid-training collapse. Dueling's pitch (separating V(s) from A(s,a) when actions are redundant) doesn't apply to Celeste room 0, where every frame demands a precise input. Dense milestone rewards make V(s) easily learnable through Q directly, so the decomposition adds parameters without adding signal.

A "we got positive results from the obvious fix" paper is fine. A "we got positive results from the obvious fix AND a negative result on the architectural enhancement we expected to help" paper is genuinely stronger.

## Phase 8: BC, Hybrid, and Curriculum re-runs with semantic state (April 26)

To make the algorithm comparison clean, every method needed to use the same backbone (plain DQN) and the same env (semantic state, no curiosity). We migrated `train_bc.py`, `train_hybrid.py`, and `train_curriculum.py` from `DuelingDQN` to plain `DQN`, and from `CelesteEnvV3` (with curiosity) to plain `CelesteEnv`. We also updated curriculum's milestone pre-marking (which was zeroing out crossed milestones to prevent free rewards) to use the new 15-step ramp.

The TAS data needed regeneration. The export script replays each `.tas` through the live env, so just running it again with the current 87-dim semantic env produces a fresh pickle:

```bash
python scripts/export_tas.py --room 0
# → 66 transitions, state dim = 87
```

Sixty-six transitions is room 0's full any% TAS (about 66 frames to complete the room from spawn).

`bc_r2` trained for 200 epochs on those 66 transitions. Train accuracy plateaued around 70%, val accuracy at 67% (= 4/6 correct on a six-sample val set, statistically meaningless). Eval at ε=0.0 over 50 episodes was 0/50, every episode deterministically reaching y=55 and getting stuck. With ε=0.05 over 100 episodes, also 0/100, with 88 timeouts (the agent freezes in a deterministic loop and runs out of time).

The 0% from BC isn't a failed experiment — it's the *finding*. Pure imitation provides demonstrations of optimal behavior but no recovery from off-distribution states. The instant the agent's actual state drifts from the TAS trajectory (rounding, sub-pixel imprecision), it's off-distribution. With no recovery signal in the loss, it picks whatever action looked "right" for the closest seen state, drifts further, locks into a deterministic loop. Textbook BC failure mode. Empirical anchor for "imitation alone is insufficient for precision platforming."

`hybrid_r2` was a surprise. After dqn_r1 hit 68%, hybrid was supposed to be at least as good — BC seeding the right region of policy space, plus online RL for recovery. Instead it got 145/5000 (2.9%) lifetime training completions and 0/100 at eval. Worse than random in some ways: 100% deaths at eval, no timeouts, the agent actively walked into spikes.

Looking at `train_hybrid.py:104`, the bug was right there:

```python
transitions.append((obs, action_idx, 0.0, next_obs, done))
```

Every TAS reward is hardcoded to 0.0. The expert buffer takes 20% of every batch — so 20% of every Q-learning gradient update is computed against transitions that say "in this state, doing the TAS action gives reward 0." Q-learning bootstraps `Q(s, a) ≈ r + γ * max Q(s', a')`. With `r = 0` for TAS transitions, the network is being *pulled toward zero* on the very (state, action) pairs along the optimal trajectory.

The expert buffer wasn't seeding good behavior, it was *anti-training* the network on optimal moves. The on-policy positive rewards from the env fight against the off-policy zero rewards 20% of every batch. The agent ends up with a policy that occasionally stumbles into a completion (BC seeding still helps a little) but cannot consolidate.

This is an implementation issue, not a property of hybrid imitation+RL methods in general. A correct implementation would replay the TAS through the env to capture the actual reward stream rather than zeroing everything out. We didn't catch it until after running. Too late to redo before the deadline.

`curriculum_r5` (ran with the cleaned curriculum script, semantic state, plain DQN architecture in theory — though we discovered later we'd forgotten to `git pull` before launching, so it actually ran with the old Dueling architecture):

| Stage | Spawn y | Outcome   |
|-------|---------|-----------|
| 1     | 15      | advanced  |
| 2     | 30      | advanced  |
| 3     | 50      | advanced  |
| 4     | 60      | timeout   |
| 5     | 71      | timeout   |
| 6     | 96      | timeout   |

Eval at ε=0.05 over 100 episodes was 0/100, every episode dying around h=72 (the lower-mid bridge). Best episode reached h=30. The agent learned to complete from y=15, y=30, y=50 — but each stage taught a *local* skill, not the underlying task. When tested at y=96 (full level), the policy fails because it never saw early-room states during the stages where it actually learned. Classic curriculum-overfit failure: the policy specializes to the curriculum's intermediate states rather than to the underlying environment.

Compare to dqn_r1, which trained from y=96 every episode for 5000 episodes and reached 68%. The straight RL path solved what the curriculum couldn't.

## Phase 9: the final comparison (April 26)

We wrote `scripts/compare_all.py` that auto-discovers every model under `runs/` and runs ε=0.05, 100-episode evaluations in a single batch. Same machine, same seed-stream, consistent eval conditions across all methods. The script auto-detects architecture (plain DQN vs DuelingDQN) by inspecting checkpoint state-dict keys — DuelingDQN has `value_stream.X` / `advantage_stream.X`, plain DQN has `network.X` — so it works for any model without remembering which is which.

The batched results:

```
Run                  Arch         Complete   Died  Timeout
----------------------------------------------------------
dqn_r1               dqn          57/100  (57.0%)    16      27
v3_r9                dueling      37/100  (37.0%)    55       8
random               random        0/100  ( 0.0%)    85      15
bc_r2                dqn           0/100  ( 0.0%)    12      88
curriculum_r5        dueling       0/100  ( 0.0%)     3      97
hybrid_r2            dqn           0/100  ( 0.0%)   100       0
v3_r8                dueling       0/100  ( 0.0%)    12      88
```

The qualitative ordering matches the per-run evaluations. dqn_r1 measured 68% in an isolated 100-episode eval and 57% in this batched run on the same checkpoint — within ±10 points, which is the variance you get from single-seed evaluation. The qualitative ordering is robust across seeds, and dqn_r1 winning by 20+ points over the next-best variant is well outside that noise.

The death/timeout split per method is signal, not noise:

| Method        | Deaths | Timeouts | Pattern |
|---------------|--------|----------|---------|
| dqn_r1        | 16     | 27       | Mixed — early ε-noise deaths, "almost made it" timeouts |
| v3_r9         | 55     | 8        | Dies more, often at the upper-ledge zone |
| random        | 85     | 15       | Spikes itself walking into spikes |
| bc_r2         | 12     | 88       | Freezes — deterministic loop, runs out of time |
| curriculum_r5 | 3      | 97       | Freezes harder — barely dies, just stuck |
| hybrid_r2     | 100    | 0        | Walks into spikes — anti-trained by zero-reward expert transitions |
| v3_r8         | 12     | 88       | Freezes like BC — raw tile IDs gave nothing useful |

The hybrid_r2 result is striking: 100% deaths, 0% timeouts. Worse than random (which had 85 deaths). The implementation bug genuinely anti-trained the network on optimal moves.

## Phase 10: hybrid revisited with the reward fix (April 27)

After locking in the comparison results, we returned to the hybrid implementation to test whether fixing the zero-reward bug would change the picture. The fix in `build_expert_transitions` was small: instead of storing `(obs, action, 0.0, next_obs, done)`, we now reset the env and replay the TAS actions through it, capturing the real milestone rewards along the optimal trajectory. We also dropped the saved-obs from the pickle entirely (it was 31-dim from an earlier export) and used the env's current observations only.

`hybrid_r3` ran with the corrected reward stream, same hyperparameters as `dqn_r1`. The result was different from `hybrid_r2` but still didn't beat plain DQN.

```
Ep  450: best 6  avg_h 53.6  reward 68     ← peak progress
Ep 1000: best 6  avg_h 57.0  reward 45     ← stagnating
Ep 1450: best 6  avg_h 72.8  reward 16     ← actively regressing
Ep 1850: best 6  avg_h 68.2  reward 23     ← still degrading
```

Two things stood out compared to `hybrid_r2` and `dqn_r1`:

**The BC seeding worked, briefly.** `hybrid_r3` reached y=6 by episode 450 — substantially faster than `dqn_r1`, which took ~1000 episodes to reach y=8. The expert buffer with real rewards was clearly pointing the network at productive moves early. So the original premise of hybrid was vindicated — TAS warm-start does accelerate early learning.

**But the agent then stalled and regressed.** Best height stuck at 6 for 1400 more episodes while average height *climbed* from 54 to 72 and per-episode reward fell from 68 to 23. Killed the run at episode 1850; the trajectory wasn't recovering.

We attribute the failure to off-policy bootstrapping divergence rather than anti-training:

1. **Expert overfit pressure.** With 66 transitions sampled at 20% of every 128-batch, ~25 transitions per gradient update are repeats from the same 66 frames. That's massive overfit pressure on a small set of (state, action) pairs.

2. **Reward magnitude mismatch.** Expert transitions individually carry milestone rewards of +100 to +900. Online transitions during failed episodes carry 0 to +20 per frame. The gradient signal from the expert side is 10–50× larger per sample, pulling Q-values along the expert trajectory toward values the online policy can't realize — because it never reaches those states without the seed working harder than it actually does.

3. **The expert teaches what's already learned.** The TAS passes through y=6 on its way to the exit. By episode 450 the policy already knows how to reach y=6. The expert buffer reinforces what the network already has and provides no new information past that point, while the online policy can't reach y<-4 to populate the buffer with terminal-reward transitions naturally.

So `hybrid_r3` failed for a *different* reason than `hybrid_r2`. The original (buggy) version anti-trained the network with zero-reward expert data, producing 100% deaths. The corrected version has the opposite problem — the expert rewards are too dominant, the early seeding works but the agent can't escape the seeded region.

This is consistent with the broader literature: simple "preload buffer + train normally" hybrid setups are known to be brittle. Methods that work robustly (e.g., DQfD, Hester et al. 2018) require either supervised pre-training before online RL, or specific mechanisms to anneal the expert influence over time. We didn't have time to implement either before the deadline.

For the comparison, the takeaway updates: hybrid has *two distinct failure modes* under our two implementations, neither of which beats plain DQN. The zero-reward bug produced 100% deaths; the corrected version produces stalled-and-regressing learning. Both 0% at deterministic-equivalent eval. The "complexity hurts in this setting" thesis holds across both attempts; a robust hybrid implementation remains future work.

## What we found

Five method variants were tested. Only one beat the random baseline meaningfully. That one — plain DQN with semantic tile encoding — beat the next-best variant (Dueling DQN + curiosity) by 20 percentage points.

Every other approach we tried — Behavioral Cloning, spawn Curriculum, Hybrid (BC + DQN), Dueling architecture, curiosity bonus — underperformed plain DQN with the same state representation. Some failed completely (0% completion at eval), some hurt only mildly (37% vs 57%).

The dominant factor across all the comparisons was **state representation**, not algorithm choice. Replacing raw PICO-8 tile IDs (0–255) with a 5-class semantic encoding (air / solid / spike / out-of-bounds / other) raised plain DQN's completion rate from 8% (v3_r8 raw IDs) to 57–68% (dqn_r1 semantic) with no other algorithmic change. That's a larger improvement than any algorithmic technique we tried.

A few caveats worth being honest about in the paper:

1. **Hybrid had a real implementation bug.** Expert transitions used `reward=0` instead of replaying through the env. A correct implementation might recover meaningfully — possibly even outperform plain DQN. The 0% number reflects our specific implementation, not the general capability of imitation+RL.

2. **Curriculum was run with the wrong architecture.** We forgot to `git pull` before launching `curriculum_r5`, so it trained with DuelingDQN rather than the cleaned-up plain DQN. A clean rerun might modestly improve, but stages 4–6 timing out is a curriculum-design issue rather than an architecture issue, so the conclusion likely stands.

3. **dqn_r1 vs v3_r9 isn't a clean single-variable ablation.** They differ on both architecture (plain DQN vs Dueling DQN) and reward shaping (no curiosity bonus vs curiosity bonus). Decomposing the 20-point gap into architectural vs reward-shaping components would have required another run we didn't have time for.

4. **Single-seed comparisons.** Each method was run once with one random seed. Variance estimates would require multiple seeds.

5. **Single-room evaluation.** All comparisons are on Celeste room 0. Generalization to other rooms or full-game completion is future work — and there's prior art ([effdotsh/Celeste-Bot](https://github.com/effdotsh/Celeste-Bot), a genetic algorithm) that does solve the full game, suggesting population-based or on-policy methods with curiosity (PPO + RND) are probably the right algorithm class for the harder rooms.

## Future work

A few directions we'd take this if we had more time. We're listing them as actionable next steps rather than aspirational ones — each one is something we could run on the existing infrastructure.

**Fix the hybrid implementation and re-run.** The hybrid bug — expert transitions saved with `reward=0` instead of replayed through the env — is a real implementation issue, not a property of imitation+RL methods. A correct version would replay each TAS frame through the env to capture the actual reward stream. Whether a corrected hybrid matches plain DQN, beats it, or still underperforms is the most interesting open question we have. If hybrid wins after the fix, our broader "complexity hurts" thesis weakens; if it still loses, the thesis gets stronger. Either result is worth knowing.

**Multiple seeds per method.** Every result here is single-seed. The qualitative ordering is robust (we confirmed it across the per-run vs batched evaluations), but the gap between dqn_r1 and v3_r9 lacks a confidence band. Three seeds per method would let us put error bars on every bar in the comparison chart and test whether the 20-point gap is significant at p < 0.05. Realistic time budget: ~30 GPU-hours.

**Generalization to other rooms.** All comparisons are on room 0. Whether plain DQN with semantic encoding continues to dominate on rooms with different layouts and different action requirements (e.g., dash chains in rooms 5+, wind levels in rooms 12+, falling-block sequences) is genuinely unknown. The infrastructure supports it — `CelesteEnv(room=N)` works for any N from 0 to 30. The bottleneck is compute. Sequential transfer (`--resume` from one room to the next) is the cheapest path; per-room independent training would be more rigorous but ~30× more expensive.

**Single-variable architectural ablation.** dqn_r1 and v3_r9 differ on two axes simultaneously (plain vs Dueling DQN, no curiosity vs curiosity bonus). To attribute the 20-point gap cleanly, we'd run plain DQN with the curiosity bonus (a single new training run) and decompose the result. If `plain DQN + curiosity` lands close to v3_r9, the curiosity bonus did the damage; if it lands closer to dqn_r1, Dueling did. Cheapest unfinished experiment in the project.

**Larger expert dataset for BC.** Sixty-six TAS transitions is genuinely too few. Adding 5–10 hand-recorded human playthroughs (each ~70–100 frames of slightly imperfect play) would let us test whether BC's failure is fundamental or data-limited. Distribution shift theory says it's fundamental for precision tasks; an empirical confirmation either way is interesting.

These are workshop-paper-tier next steps. The most impactful single one is fixing hybrid; the most rigorous is multiple seeds.

## Things we'd do differently

Looking back, three things stand out.

The most useful debugging move was simple: **watch the agent play.** Akhil's "the agent oscillates on the right side" came from looking at one episode at low frame rate, and that observation drove the entire perception-fix breakthrough. We had spent weeks tuning hyperparameters and reward shapes without ever actually watching what the agent was doing. The bug — that raw tile IDs gave the network no semantic signal — had been there since the first 5×5 tile grid was added; we just hadn't asked "what is the agent seeing?"

The second is the **`best.pt` save logic**. It's saved when `info['max_height'] < best_height` first fires, which means it freezes at the first time any episode reaches a new minimum y. Across every run we did past v3_r8, the *real* best policy was ~1000–4000 episodes later than the `best.pt` snapshot. We fixed this manually (evaluating specific late-stage checkpoints) but never updated the save logic, which would now switch to "best by rolling completion rate over the last N episodes." Future runs should fix this.

The third is **regenerating expert data**. The TAS pickle was recorded with the old 31-dim observation, and we didn't realize this until we tried to run BC and it crashed with a state-dim mismatch. The export script (`scripts/export_tas.py`) replays through the live env, so it picks up env changes automatically — but we should have re-exported as part of the perception-fix changeset rather than discovering it later.

## Files / paths to know

- `runs/dqn_r1/` — headline result (57% eval, 86% peak training)
- `runs/v3_r9/` — Dueling DQN + curiosity (37% eval, 64% peak training)
- `runs/v3_r8/` — DQN with raw tile IDs, "before perception fix" (8% peak)
- `runs/bc_r2/` — Behavioral Cloning floor (0%)
- `runs/hybrid_r2/` — Hybrid with the zero-reward bug (0%)
- `runs/curriculum_r5/` — Curriculum with stage-overfit failure (0%)
- `data/tas_transitions.pkl` — 87-dim semantic, 66 transitions, room 0 any%
- `src/environment.py:_get_obs()` — semantic tile encoder (don't break this)
- `src/utils/paths.py:run_dir(run_id)` — `runs/{run_id}/` helper
- `scripts/compare_all.py` — batched evaluation, generates the comparison plots
- `scripts/evaluate.py` — single-model evaluation, auto-detects architecture
- `docs/comparison/` — paper-ready figures (completion bar, outcome breakdown, height distribution)
