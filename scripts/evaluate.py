"""
Evaluate trained agent performance.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/dqn_best.pt --episodes 100
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent


def evaluate_agent(model_path: str, room: int = 0, num_episodes: int = 100):
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
    
    agent.epsilon = 0.0
    
    rewards = []
    heights = []
    successes = 0
    deaths = 0
    
    print(f"\nEvaluating on room {room} for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        heights.append(info['max_height'])
        
        if info.get('player_y', 999) < -8:
            successes += 1
        elif not info['player_alive']:
            deaths += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} - Success rate: {successes/(episode+1)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Room: {room}")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Death Rate: {deaths}/{num_episodes} ({100*deaths/num_episodes:.1f}%)")
    print("-" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Height: {np.mean(heights):.1f}")
    print(f"Best Height: {min(heights):.1f}")
    print("=" * 60)


def evaluate_random_baseline(room: int = 0, num_episodes: int = 100):
    """Evaluate random baseline for comparison."""
    
    env = CelesteEnv(room=room, max_steps=1000, use_simple_actions=True)
    
    rewards = []
    heights = []
    successes = 0
    
    print(f"\nRandom baseline on room {room}...")
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        heights.append(info['max_height'])
        if info.get('player_y', 999) < -8:
            successes += 1
    
    print(f"\nRandom Baseline:")
    print(f"  Success Rate: {100*successes/num_episodes:.1f}%")
    print(f"  Mean Reward: {np.mean(rewards):.2f}")
    print(f"  Best Height: {min(heights):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agent")
    parser.add_argument("--model", type=str, default="models/dqn_best.pt")
    parser.add_argument("--room", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--baseline", action="store_true", help="Also run random baseline")
    
    args = parser.parse_args()
    
    evaluate_agent(args.model, args.room, args.episodes)
    
    if args.baseline:
        evaluate_random_baseline(args.room, args.episodes)
