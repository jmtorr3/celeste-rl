"""
Evaluate a trained agent on Celeste room 0.

Architecture (DQN vs DuelingDQN) is auto-detected from the checkpoint.
Use --dueling / --no-dueling to override.

Usage:
    python scripts/evaluate.py --run-id v3_r9 --epsilon 0.05
    python scripts/evaluate.py --run-id dqn_r1 --epsilon 0.05 --episodes 100
    python scripts/evaluate.py --model runs/v3_r9/checkpoint_ep1950.pt --epsilon 0.05
    python scripts/evaluate.py --baseline-only                  # random-action baseline only
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent
from src.network import DuelingDQN


def detect_architecture(model_path):
    """Inspect a checkpoint and decide whether it was saved by DuelingDQN or plain DQN.

    Returns 'dueling' or 'dqn'. Raises if neither matches.
    """
    ckpt = torch.load(model_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'policy_net' in ckpt:
        sd = ckpt['policy_net']
    else:
        sd = ckpt
    keys = list(sd.keys())
    if any('value_stream' in k or 'advantage_stream' in k for k in keys):
        return 'dueling'
    if any(k.startswith('network.') for k in keys):
        return 'dqn'
    raise ValueError(f"Cannot detect architecture from keys: {keys[:4]}...")


def evaluate_agent(model_path, room=0, num_episodes=100, epsilon=0.0, dueling=None, max_steps=1000):
    """Evaluate a trained model. If dueling=None, auto-detect architecture from the checkpoint."""
    env = CelesteEnv(room=room, max_steps=max_steps)

    if dueling is None:
        try:
            arch = detect_architecture(model_path)
            dueling = (arch == 'dueling')
            print(f"Auto-detected architecture: {arch}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: {e}. Defaulting to plain DQN.")
            dueling = False

    agent = DQNAgent(
        state_dim=env._get_obs_dim(),
        action_dim=env.n_actions,
        device="cpu",
        network_cls=DuelingDQN if dueling else None,
    )

    try:
        agent.load(model_path)
        print(f"Loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        return

    agent.epsilon = epsilon
    print(f"Epsilon:    {epsilon}")
    print(f"Architecture: {'DuelingDQN' if dueling else 'DQN'}")
    print(f"Episodes:   {num_episodes}")
    print(f"Room:       {room}")
    print("-" * 60)

    rewards = []
    heights = []
    successes = 0
    deaths = 0
    timeouts = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0

        while True:
            # training=True so epsilon is applied; greedy when epsilon=0
            action = agent.select_action(state, training=(epsilon > 0))
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        rewards.append(episode_reward)
        heights.append(info['max_height'])

        if info.get('completed', False):
            successes += 1
            status = "COMPLETE"
        elif not info['player_alive']:
            deaths += 1
            status = f"died at h={info['max_height']:.0f}"
        else:
            timeouts += 1
            status = f"timeout at h={info['max_height']:.0f}"

        print(f"  Ep {episode + 1:>3}: reward={episode_reward:>8.2f}  {status}")

    print("=" * 60)
    print(f"Success rate:  {successes}/{num_episodes} ({100 * successes / num_episodes:.1f}%)")
    print(f"Death rate:    {deaths}/{num_episodes} ({100 * deaths / num_episodes:.1f}%)")
    print(f"Timeout rate:  {timeouts}/{num_episodes} ({100 * timeouts / num_episodes:.1f}%)")
    print("-" * 60)
    print(f"Mean reward:   {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean height:   {np.mean(heights):.1f}")
    print(f"Best height:   {min(heights):.1f}")
    print(f"Worst height:  {max(heights):.1f}")
    print(f"Unique heights observed: {len(set(heights))}")
    print("=" * 60)


def evaluate_random_baseline(room=0, num_episodes=100, max_steps=1000):
    env = CelesteEnv(room=room, max_steps=max_steps)

    rewards = []
    heights = []
    successes = 0

    print(f"\nRandom baseline on room {room} ({num_episodes} episodes)...")

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        while True:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        heights.append(info['max_height'])
        if info.get('completed', False):
            successes += 1

    print(f"  Success rate: {100 * successes / num_episodes:.1f}%")
    print(f"  Mean reward:  {np.mean(rewards):.2f}")
    print(f"  Best height:  {min(heights):.1f}")


def resolve_model_path(args):
    if args.model:
        return args.model
    if args.run_id:
        return os.path.join("runs", args.run_id, "best.pt")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Celeste agent")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID — resolves to runs/{run_id}/best.pt")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit model path (overrides --run-id)")
    parser.add_argument("--room", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Exploration rate during eval (0.0 = deterministic)")
    parser.add_argument("--dueling", action="store_true", default=None,
                        help="Force DuelingDQN architecture. Default: auto-detect from checkpoint.")
    parser.add_argument("--no-dueling", dest="dueling", action="store_false",
                        help="Force plain DQN architecture (overrides auto-detect).")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--baseline", action="store_true",
                        help="Also run a random-action baseline for comparison")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip the model and only run the random baseline")
    args = parser.parse_args()

    if not args.baseline_only:
        model_path = resolve_model_path(args)
        if model_path is None:
            parser.error("Must provide --run-id or --model (or use --baseline-only).")
        evaluate_agent(
            model_path,
            room=args.room,
            num_episodes=args.episodes,
            epsilon=args.epsilon,
            dueling=args.dueling,
            max_steps=args.max_steps,
        )

    if args.baseline or args.baseline_only:
        evaluate_random_baseline(args.room, args.episodes, args.max_steps)
