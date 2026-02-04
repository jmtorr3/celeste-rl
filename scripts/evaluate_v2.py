"""
Evaluate trained agent with optional exploration.

Usage:
    python scripts/evaluate_v2.py --model models/dqn_best.pt
    python scripts/evaluate_v2.py --model models/dqn_best.pt --epsilon 0.05
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent


def evaluate_agent(model_path: str, room: int = 0, num_episodes: int = 100, epsilon: float = 0.0):
    """Evaluate agent performance."""
    
    env = CelesteEnv(room=room, max_steps=1000, use_simple_actions=True)
    
    agent = DQNAgent(
        state_dim=env._get_obs_dim(),
        action_dim=env.n_actions,
        device="cpu"
    )
    
    try:
        agent.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        return
    
    # Set epsilon for evaluation (0 = deterministic, >0 = some randomness)
    agent.epsilon = epsilon
    print(f"  Evaluation epsilon: {epsilon}")
    
    rewards = []
    heights = []
    successes = 0
    deaths = 0
    
    print(f"\nEvaluating on room {room} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Use training=True so epsilon is applied
            action = agent.select_action(state, training=(epsilon > 0))
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        heights.append(info['max_height'])
        
        if info.get('player_y', 999) < -8:
            successes += 1
            status = "✓ COMPLETE"
        elif not info['player_alive']:
            deaths += 1
            status = f"✗ died at height={info['max_height']:.0f}"
        else:
            status = f"  timeout at height={info['max_height']:.0f}"
        
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, {status}")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Room: {room}")
    print(f"Episodes: {num_episodes}")
    print(f"Epsilon: {epsilon}")
    print("-" * 60)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Death Rate: {deaths}/{num_episodes} ({100*deaths/num_episodes:.1f}%)")
    print(f"Timeout Rate: {num_episodes - successes - deaths}/{num_episodes}")
    print("-" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Height: {np.mean(heights):.1f} ± {np.std(heights):.1f}")
    print(f"Best Height: {min(heights):.1f}")
    print(f"Worst Height: {max(heights):.1f}")
    print("-" * 60)
    print(f"Unique outcomes: {len(set(heights))}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent")
    parser.add_argument("--model", type=str, default="models/dqn_best.pt")
    parser.add_argument("--room", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate during eval")
    
    args = parser.parse_args()
    
    evaluate_agent(args.model, args.room, args.episodes, args.epsilon)
