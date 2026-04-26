"""
Curriculum Training — DQN with backwards curriculum over spawn positions.

Starts the agent near the exit (y=15) where only 1-2 moves are needed to
complete the level. Once it achieves >= advance_threshold completion rate
over the last advance_window episodes, the spawn point moves down toward
the full level start (y=96). Agent weights carry over between stages.

Stages:
  1. y=15, 100 steps  — exit approach
  2. y=30, 150 steps  — upper section
  3. y=50, 200 steps  — mid-level
  4. y=70, 300 steps  — lower-mid
  5. y=96, 500 steps  — full level (normal start)

Usage:
    python src/train_curriculum.py --run-id curriculum_r1 --device cuda
    python src/train_curriculum.py --eval-only --model models/curriculum_r1_best.pt
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.train_v3 import CelesteEnvV3
from src.agent import DQNAgent
from src.environment import CelesteEnv
from src.network import DuelingDQN


# (spawn_y, spawn_x, max_steps, label)
# x positions sampled from TAS replay at the corresponding y — ensures the
# agent spawns on the actual level path, not inside a wall or under a ceiling.
STAGES = [
    (15,  97,  100, "Stage 1 — y=15  (exit approach)"),
    (30,  89,  150, "Stage 2 — y=30  (upper section)"),
    (50,  79,  200, "Stage 3 — y=50  (mid-level)"),
    (60,  89,  250, "Stage 4 — y=60  (mid-lower bridge)"),
    (71,  88,  300, "Stage 5 — y=71  (lower-mid)"),
    (96,  None, 500, "Stage 6 — y=96  (full level)"),  # normal spawn, no teleport
]


class CurriculumEnv(CelesteEnvV3):
    """
    CelesteEnvV3 with a configurable spawn position.

    After the normal reset, teleport the player to (spawn_x, spawn_y).
    spawn_x should come from TAS data so the player lands on the actual
    level path, not inside a wall or under a ceiling.
    If spawn_x is None, only y is overridden (x stays at game default).
    Milestones above the spawn point are pre-marked to avoid free rewards.
    """

    def __init__(self, spawn_y: int = 96, spawn_x: int = None, **kwargs):
        super().__init__(**kwargs)
        self.spawn_y = spawn_y
        self.spawn_x = spawn_x

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        player = self._get_player()
        if player is not None:
            player.y = self.spawn_y
            if self.spawn_x is not None:
                player.x = self.spawn_x
            self.prev_y = player.y
            self.prev_x = player.x
            self.best_height_this_episode = player.y
            self.max_height_reached = player.y
            # pre-mark milestones the agent already starts above
            for threshold in (40, 20, 10, 0, -5):
                if self.spawn_y < threshold:
                    self.milestones_hit.add(threshold)
        return self._get_obs(), self._get_info()


def make_agent(args, device):
    dummy_env = CelesteEnv()
    return DQNAgent(
        state_dim=(dummy_env._get_obs_dim()),
        action_dim=dummy_env.n_actions,
        lr=args.lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=args.epsilon_end,
        epsilon_decay=1.0,   # disabled — curriculum manages epsilon per episode
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=device,
        network_cls=DuelingDQN,
    )


def train_stage(agent, env, stage_label, max_episodes, advance_window,
                advance_threshold, epsilon_decay, log_interval, run_id, save_dir, global_ep):
    """
    Train one curriculum stage.
    Returns (global_ep, completed_stage, heights, rewards, completions_log).
    completed_stage=True if advancement criterion was met.
    """
    heights = []
    rewards = []
    completions_log = []
    best_height = float('inf')
    stage_completions = 0

    print(f"\n{'='*60}")
    spawn_info = f"spawn=({env.spawn_x},{env.spawn_y})" if env.spawn_x else f"spawn_y={env.spawn_y}"
    print(f"{stage_label}  |  {spawn_info}  max_steps={env.max_steps}")
    print(f"Advance when {advance_threshold:.0%} completion over last {advance_window} eps")
    print(f"Max episodes this stage: {max_episodes}")
    print(f"{'='*60}")

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, terminated or truncated)
            agent.update()
            episode_reward += reward
            state = next_state
            if terminated or truncated:
                break

        completed = info.get('completed', False)
        heights.append(info['max_height'])
        rewards.append(episode_reward)
        completions_log.append(completed)
        if completed:
            stage_completions += 1

        if info['max_height'] < best_height:
            best_height = info['max_height']
            agent.save(str(save_dir / f'{run_id}_best.pt'))

        # Decay epsilon once per episode
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * epsilon_decay)

        global_ep += 1

        if ep % log_interval == 0:
            window_h = heights[-log_interval:]
            window_r = rewards[-log_interval:]
            recent = sum(completions_log[-log_interval:])
            print(
                f"  Ep {ep:>5} (total {global_ep:>6}) | "
                f"reward {np.mean(window_r):>7.1f} | "
                f"height {np.mean(window_h):>6.1f} | "
                f"best {best_height:>6.1f} | "
                f"complete {recent}/{log_interval} | "
                f"eps {agent.epsilon:.3f}"
            )

        # Check advancement criterion
        if len(completions_log) >= advance_window:
            recent_rate = sum(completions_log[-advance_window:]) / advance_window
            if recent_rate >= advance_threshold:
                print(f"\n  >> STAGE COMPLETE — {recent_rate:.0%} completion over last {advance_window} eps <<")
                return global_ep, True, heights, rewards, completions_log

    print(f"\n  >> Stage timeout ({max_episodes} eps) — completions: {stage_completions}/{max_episodes} <<")
    return global_ep, False, heights, rewards, completions_log


def train(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)

    agent = make_agent(args, device)

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    print(f"Run ID: {args.run_id}")

    global_ep = 0
    stage_results = []
    all_rewards = []
    all_heights = []
    all_completions = []
    stage_boundaries = []  # list of (cumulative_episode_index, label) at each stage start

    stages = STAGES[args.start_stage - 1:]  # allow starting mid-curriculum

    for i, (spawn_y, spawn_x, max_steps, label) in enumerate(stages):
        stage_num = args.start_stage + i

        # Mark the cumulative episode index where this stage begins
        stage_boundaries.append((len(all_rewards), f'S{stage_num} y={spawn_y}'))

        env = CurriculumEnv(
            spawn_y=spawn_y,
            spawn_x=spawn_x,
            room=args.room,
            max_steps=max_steps,
        )

        global_ep, advanced, stage_heights, stage_rewards, stage_completions = train_stage(
            agent=agent,
            env=env,
            stage_label=label,
            max_episodes=args.max_episodes_per_stage,
            advance_window=args.advance_window,
            advance_threshold=args.advance_threshold,
            epsilon_decay=args.epsilon_decay,
            log_interval=args.log_interval,
            run_id=args.run_id,
            save_dir=save_dir,
            global_ep=global_ep,
        )

        all_rewards.extend(stage_rewards)
        all_heights.extend(stage_heights)
        all_completions.extend(stage_completions)

        agent.save(str(save_dir / f'{args.run_id}_stage{stage_num}.pt'))
        stage_results.append((stage_num, spawn_y, advanced))

        # Reset epsilon on every stage change so the agent re-explores from the new spawn
        if stage_num < len(STAGES):
            reset_eps = args.epsilon_on_advance if advanced else min(1.0, args.epsilon_on_advance * 1.4)
            agent.epsilon = max(args.epsilon_end, reset_eps)
            print(f"  Epsilon reset to {agent.epsilon:.3f} for next stage ({'advanced' if advanced else 'timeout'})")

    agent.save(str(save_dir / f'{args.run_id}_final.pt'))

    print(f"\n{'='*60}")
    print(f"CURRICULUM TRAINING COMPLETE  |  {global_ep} total episodes")
    print(f"{'='*60}")
    for stage_num, spawn_y, advanced in stage_results:
        status = "advanced" if advanced else "timeout"
        print(f"  Stage {stage_num} (spawn_y={spawn_y}): {status}")

    with open(docs_dir / f'training_{args.run_id}.pkl', 'wb') as f:
        pickle.dump({
            'stage_results': stage_results,
            'total_episodes': global_ep,
            'rewards': all_rewards,
            'heights': all_heights,
            'completions_log': all_completions,
            'stage_boundaries': stage_boundaries,
        }, f)

    from src.utils.plot import plot_run
    plot_run(
        args.run_id, all_rewards, all_heights,
        completions=all_completions,
        stage_boundaries=stage_boundaries,
        save_dir=str(docs_dir),
    )


def evaluate(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = CelesteEnv(room=args.room, max_steps=500)

    agent = DQNAgent(
        state_dim=env._get_obs_dim(),
        action_dim=env.n_actions,
        device=device,
        network_cls=DuelingDQN,
    )
    model_path = args.model or str(Path(args.save_dir) / f'{args.run_id}_best.pt')
    agent.load(model_path)
    agent.epsilon = 0.0
    print(f"Loaded {model_path}")

    completions = 0
    heights = []

    print("=" * 60)
    print(f"EVALUATION  |  {args.eval_episodes} episodes  room={args.room}  (full level, spawn_y=96)")
    print("=" * 60)

    for ep in range(args.eval_episodes):
        state, _ = env.reset()
        while True:
            action = agent.select_action(state, training=False)
            state, _, done, trunc, info = env.step(action)
            if done or trunc:
                break
        heights.append(info['max_height'])
        if info.get('completed', False):
            completions += 1
            print(f"  Ep {ep+1:>3}: COMPLETE")
        else:
            print(f"  Ep {ep+1:>3}: height={info['max_height']:.0f}")

    print("-" * 60)
    print(f"Success rate: {completions}/{args.eval_episodes} ({100*completions/args.eval_episodes:.0f}%)")
    print(f"Mean height:  {np.mean(heights):.1f}")
    print(f"Best height:  {min(heights):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Celeste Curriculum Training')
    parser.add_argument('--run-id', type=str, default='curriculum',
                        help='Prefix for all saved files')
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epsilon-decay', type=float, default=0.99700,
                        help='Per-episode decay — reaches floor (~0.10) in ~768 eps per stage')
    parser.add_argument('--epsilon-end', type=float, default=0.10)
    parser.add_argument('--epsilon-on-advance', type=float, default=0.50,
                        help='Epsilon to reset to when advancing to next stage')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--max-episodes-per-stage', type=int, default=2000,
                        help='Give up on a stage after this many episodes')
    parser.add_argument('--advance-window', type=int, default=100,
                        help='Episodes window for advancement criterion')
    parser.add_argument('--advance-threshold', type=float, default=0.50,
                        help='Completion rate required to advance to next stage')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='Start from this stage (1=exit, 6=full level)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a checkpoint (implies --start-stage)')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args)
    else:
        train(args)
        evaluate(args)


if __name__ == '__main__':
    main()
