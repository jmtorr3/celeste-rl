"""
Evaluate a trained agent on Celeste room 0.

Usage:
    python scripts/evaluate.py --run-id v3_r9 --epsilon 0.05 --dueling
    python scripts/evaluate.py --model runs/v3_r9/checkpoint_ep1950.pt --episodes 100 --epsilon 0.05 --dueling
    python scripts/evaluate.py --baseline                       # random-action baseline only
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent
from src.network import DuelingDQN


def evaluate_agent(model_path, room=0, num_episodes=100, epsilon=0.0, dueling=False, max_steps=1000):
    env = CelesteEnv(room=room, max_steps=max_steps)

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
    parser.add_argument("--dueling", action="store_true",
                        help="Use DuelingDQN architecture (required for v3+/hybrid/curriculum models)")
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
