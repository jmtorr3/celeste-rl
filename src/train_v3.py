"""
Training V3 - Professor Feedback Fixes + Exploration Bonuses

Changes from v2:
  - State: 31-dim (x/128, y/128, spd_x, spd_y, grace, djump, 5x5 tile grid)
  - Uses CelesteEnv from environment.py (fixed coordinate normalization)
  - Exploration bonuses layered on top via subclass

Usage:
    python src/train_v3.py
    python src/train_v3.py --episodes 5000 --device cuda
    python src/train_v3.py --resume models/v3_checkpoint.pt
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent
from src.network import DuelingDQN
import torch


class CelesteEnvV3(CelesteEnv):
    """
    CelesteEnv with per-episode and global position exploration bonuses.
    Inherits the fixed 31-dim observation space from environment.py.
    """

    def reset(self, seed=None, options=None):
        self.visited_positions = set()
        self.global_visit_counts = getattr(self, 'global_visit_counts', {})
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        player_x = info.get('player_x', 0)
        player_y = info.get('player_y', 0)

        if info.get('player_alive', False):
            pos = (int(player_x // 4), int(player_y // 4))

            if pos not in self.visited_positions:
                self.visited_positions.add(pos)
                reward += 0.5

                count = self.global_visit_counts.get(pos, 0) + 1
                self.global_visit_counts[pos] = count
                if count < 10:
                    reward += 0.2

        info['visited_count'] = len(self.visited_positions)
        info['global_positions'] = len(self.global_visit_counts)
        return obs, reward, terminated, truncated, info


def train(args):
    env = CelesteEnvV3(room=args.room, max_steps=args.max_steps)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    state_dim = env._get_obs_dim()  # 31
    action_dim = env.n_actions

    print(f"State dim:  {state_dim}")
    print(f"Action dim: {action_dim}")

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        device=device,
        network_cls=DuelingDQN,
    )

    start_episode = 0
    if args.resume:
        agent.load(args.resume)
        # Infer episode from checkpoint name if possible
        ckpt_name = Path(args.resume).stem  # e.g. "v3_checkpoint_ep1000"
        parts = ckpt_name.split('_ep')
        if len(parts) == 2 and parts[1].isdigit():
            start_episode = int(parts[1])
        print(f"Resumed from {args.resume} (episode ~{start_episode})")

    run_id = args.run_id
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    print(f"Run ID: {run_id}")

    rewards = []
    heights = []
    completions_log = []
    completions = 0
    best_height = float('inf')

    print("=" * 60)
    print(f"TRAINING V3  |  episodes={args.episodes}  max_steps={args.max_steps}")
    print("=" * 60)

    for episode in range(start_episode, start_episode + args.episodes):
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

        rewards.append(episode_reward)
        heights.append(info['max_height'])
        completed = info.get('completed', False)
        completions_log.append(completed)
        if completed:
            completions += 1

        if info['max_height'] < best_height:
            best_height = info['max_height']
            agent.save(str(save_dir / f'{run_id}_best.pt'))

        ep_num = episode + 1

        if ep_num % args.log_interval == 0:
            window = rewards[-args.log_interval:]
            h_window = heights[-args.log_interval:]
            recent_complete = sum(1 for c in completions_log[-args.log_interval:] if c)
            print(
                f"Ep {ep_num:>6} | "
                f"reward {np.mean(window):>7.1f} | "
                f"height {np.mean(h_window):>6.1f} | "
                f"best {best_height:>6.1f} | "
                f"complete {recent_complete}/{args.log_interval} | "
                f"eps {agent.epsilon:.3f} | "
                f"buf {len(agent.buffer)}"
            )

        if ep_num % args.save_interval == 0:
            ckpt_path = save_dir / f'{run_id}_checkpoint_ep{ep_num}.pt'
            agent.save(str(ckpt_path))
            print(f"  [checkpoint -> {ckpt_path}]")

    agent.save(str(save_dir / f'{run_id}_final.pt'))

    with open(docs_dir / f'training_{run_id}.pkl', 'wb') as f:
        pickle.dump({
            'rewards': rewards,
            'heights': heights,
            'completions_log': completions_log,
            'completions': completions,
        }, f)

    from src.utils.plot import plot_run
    plot_run(run_id, rewards, heights, completions=completions_log, save_dir=str(docs_dir))

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Total completions: {completions}/{args.episodes}")
    print(f"Best height:       {best_height:.1f}")
    print(f"{'='*60}")


def evaluate(args):
    env = CelesteEnvV3(room=args.room, max_steps=args.max_steps)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print(f"EVALUATION  |  {args.eval_episodes} episodes")
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
    parser = argparse.ArgumentParser(description='Celeste RL - Training V3')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epsilon-decay', type=float, default=0.9995)
    parser.add_argument('--epsilon-end', type=float, default=0.10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--model', type=str, default=None, help='Model path for eval-only')
    parser.add_argument('--run-id', type=str, default='v3', help='Prefix for all saved files — use a unique ID per run to avoid overwriting')
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args)
    else:
        train(args)
        if not args.eval_only:
            args.eval_only = True
            evaluate(args)


if __name__ == '__main__':
    main()
