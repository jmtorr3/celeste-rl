"""
Hybrid Training — DQN warm-started from TAS expert data.

Pre-fills the replay buffer with TAS (state, action, reward, next_state, done)
transitions before online training begins. A protected expert partition ensures
expert data is never displaced by online experience.

Usage:
    python src/train_hybrid.py --run-id hybrid_r1 --device cuda
    python src/train_hybrid.py --run-id hybrid_r1 --expert-fraction 0.25 --device cuda
    python src/train_hybrid.py --eval-only --model models/hybrid_r1_best.pt
"""

import argparse
import pickle
import numpy as np
from collections import deque
from pathlib import Path
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.train_v3 import CelesteEnvV3
from src.agent import DQNAgent
from src.environment import CelesteEnv
from src.network import DuelingDQN


class HybridBuffer:
    """
    Replay buffer with a protected expert partition.

    expert_capacity slots are reserved for TAS transitions and never evicted.
    online_capacity slots fill with environment experience (FIFO).
    Samples mix both partitions at the ratio expert_fraction : (1 - expert_fraction).
    """

    def __init__(self, total_capacity: int, expert_fraction: float = 0.2):
        self.expert_fraction = expert_fraction
        self.expert_capacity = int(total_capacity * expert_fraction)
        self.online_capacity = total_capacity - self.expert_capacity
        self.expert = deque(maxlen=self.expert_capacity)
        self.online = deque(maxlen=self.online_capacity)

    def push_expert(self, *transition):
        self.expert.append(transition)

    def push(self, *transition):
        self.online.append(transition)

    def sample(self, batch_size: int):
        n_expert = min(int(batch_size * self.expert_fraction), len(self.expert))
        n_online = batch_size - n_expert

        # Fall back to all-online if expert buffer not populated yet
        if len(self.online) < n_online:
            n_online = len(self.online)
            n_expert = min(batch_size - n_online, len(self.expert))

        batch = []
        if n_expert > 0:
            batch += random.sample(list(self.expert), n_expert)
        if n_online > 0:
            batch += random.sample(list(self.online), n_online)

        random.shuffle(batch)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.expert) + len(self.online)


def build_expert_transitions(tas_path: str, env: CelesteEnvV3):
    """
    Convert (obs, action_idx) TAS pairs into full (s, a, r, s', done) transitions
    by stepping through the environment.
    """
    with open(tas_path, 'rb') as f:
        tas_data = pickle.load(f)

    # tas_data is list of (obs, action_idx) — we need to re-step to get rewards
    # Group by room boundary: re-export isn't feasible here, so we construct
    # synthetic transitions: reward=0, done=False for all expert steps.
    # The agent learns WHAT to do from expert states; reward shaping during
    # online training teaches it WHY.
    transitions = []
    for i, (obs, action_idx) in enumerate(tas_data):
        if i + 1 < len(tas_data):
            next_obs = tas_data[i + 1][0]
            done = False
        else:
            next_obs = obs  # terminal — next obs doesn't matter
            done = True
        transitions.append((obs, action_idx, 0.0, next_obs, done))

    return transitions


def train(args):
    env = CelesteEnvV3(room=args.room, max_steps=args.max_steps)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    state_dim = env._get_obs_dim()
    action_dim = env.n_actions
    print(f"State dim:  {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Run ID:     {args.run_id}")

    # Build agent with HybridBuffer
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

    # Replace standard buffer with HybridBuffer
    hybrid_buf = HybridBuffer(args.buffer_size, expert_fraction=args.expert_fraction)
    agent.buffer = hybrid_buf

    # Load and push expert transitions
    print(f"\nLoading expert data from {args.data}...")
    expert_transitions = build_expert_transitions(args.data, env)
    for t in expert_transitions:
        hybrid_buf.push_expert(*t)
    print(f"Expert transitions loaded: {len(hybrid_buf.expert)} "
          f"(capacity {hybrid_buf.expert_capacity})")
    print(f"Expert fraction: {args.expert_fraction:.0%} of each batch")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)

    rewards = []
    heights = []
    completions_log = []
    completions = 0
    best_height = float('inf')

    print("=" * 60)
    print(f"TRAINING HYBRID  |  episodes={args.episodes}  expert_fraction={args.expert_fraction:.0%}")
    print("=" * 60)

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            hybrid_buf.push(state, action, reward, next_state, terminated or truncated)
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
            agent.save(str(save_dir / f'{args.run_id}_best.pt'))

        ep_num = episode + 1

        if ep_num % args.log_interval == 0:
            window = rewards[-args.log_interval:]
            h_window = heights[-args.log_interval:]
            recent_complete = sum(1 for h in h_window if h < -8)
            print(
                f"Ep {ep_num:>6} | "
                f"reward {np.mean(window):>7.1f} | "
                f"height {np.mean(h_window):>6.1f} | "
                f"best {best_height:>6.1f} | "
                f"complete {recent_complete}/{args.log_interval} | "
                f"eps {agent.epsilon:.3f} | "
                f"buf {len(hybrid_buf)}"
            )

        if ep_num % args.save_interval == 0:
            ckpt = save_dir / f'{args.run_id}_checkpoint_ep{ep_num}.pt'
            agent.save(str(ckpt))
            print(f"  [checkpoint -> {ckpt}]")

    agent.save(str(save_dir / f'{args.run_id}_final.pt'))

    with open(docs_dir / f'training_{args.run_id}.pkl', 'wb') as f:
        pickle.dump({
            'rewards': rewards,
            'heights': heights,
            'completions_log': completions_log,
            'completions': completions,
        }, f)

    from src.utils.plot import plot_run
    plot_run(args.run_id, rewards, heights, completions=completions_log, save_dir=str(docs_dir))

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Total completions: {completions}/{args.episodes}")
    print(f"Best height:       {best_height:.1f}")
    print(f"{'='*60}")


def evaluate(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    env = CelesteEnv(room=args.room, max_steps=args.max_steps)

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
    print(f"EVALUATION  |  {args.eval_episodes} episodes  room={args.room}")
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
    parser = argparse.ArgumentParser(description='Celeste Hybrid Training')
    parser.add_argument('--data', type=str, default='data/tas_transitions.pkl')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epsilon-decay', type=float, default=0.999990)
    parser.add_argument('--epsilon-end', type=float, default=0.10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--buffer-size', type=int, default=500000)
    parser.add_argument('--expert-fraction', type=float, default=0.20,
                        help='Fraction of each training batch drawn from expert data')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=500)
    parser.add_argument('--save-dir', type=str, default='models')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--run-id', type=str, default='hybrid',
                        help='Prefix for all saved files — use unique ID per run')
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
