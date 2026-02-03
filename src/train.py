"""
Training Script for Celeste RL Agent

Usage:
    python -m src.train
    python src/train.py
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Handle imports whether run as module or script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.environment import CelesteEnv
    from src.agent import DQNAgent
else:
    from .environment import CelesteEnv
    from .agent import DQNAgent

import torch


def train(
    env: CelesteEnv,
    agent: DQNAgent,
    num_episodes: int = 3000,
    max_steps: int = 500,
    log_interval: int = 50,
    save_interval: int = 200,
    save_dir: str = "models"
):
    """
    Train the DQN agent.
    
    Args:
        env: Celeste environment
        agent: DQN agent
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        log_interval: Episodes between logging
        save_interval: Episodes between saving checkpoints
        save_dir: Directory to save models
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    episode_rewards = []
    episode_heights = []
    best_reward = float('-inf')
    best_height = float('inf')
    
    print("=" * 60)
    print("TRAINING START")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_heights.append(info['max_height'])
        
        # Save best model (by height reached)
        if info['max_height'] < best_height:
            best_height = info['max_height']
            agent.save(save_path / "dqn_best.pt")
        
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Logging
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_height = np.mean(episode_heights[-log_interval:])
            min_height = min(episode_heights[-log_interval:])
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward:  {avg_reward:.2f}")
            print(f"  Avg Height:  {avg_height:.1f}")
            print(f"  Best Height: {min_height:.1f} (this batch), {best_height:.1f} (overall)")
            print(f"  Epsilon:     {agent.epsilon:.3f}")
            print(f"  Buffer:      {len(agent.buffer)}")
        
        # Regular checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save(save_path / "dqn_checkpoint.pt")
            print(f"  [Checkpoint saved]")
    
    # Save final model
    agent.save(save_path / "dqn_final.pt")
    
    return episode_rewards, episode_heights


def evaluate(env: CelesteEnv, agent: DQNAgent, num_episodes: int = 20):
    """Evaluate the trained agent."""
    rewards = []
    heights = []
    successes = 0
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
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
        
        success = info['player_y'] < -8 if info['player_alive'] else False
        if success:
            successes += 1
        
        status = "✓ COMPLETE" if success else f"height={info['max_height']:.0f}"
        print(f"Episode {episode + 1}: reward={episode_reward:.2f}, {status}")
    
    print("-" * 60)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Height: {np.mean(heights):.1f}")
    print(f"Best Height: {min(heights):.1f}")
    print("=" * 60)
    
    return rewards, heights


def plot_results(rewards, heights, save_path="docs/training_curve.png"):
    """Plot and save training curves."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rewards
        ax1.plot(rewards, alpha=0.3, label='Episode Reward')
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), ma, label=f'{window}-ep MA', linewidth=2)
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heights
        ax2.plot([96 - h for h in heights], alpha=0.3, label='Height Reached')
        if len(heights) >= window:
            ma = np.convolve([96 - h for h in heights], np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(heights)), ma, label=f'{window}-ep MA', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Height Reached (from start)')
        ax2.set_title('Progress Up the Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved training curve to {save_path}")
        
    except ImportError:
        print("matplotlib not installed, skipping plot")


def main():
    parser = argparse.ArgumentParser(description="Train Celeste RL Agent")
    parser.add_argument("--episodes", type=int, default=3000, help="Training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--room", type=int, default=0, help="Room number (0-30)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate")
    parser.add_argument("--model", type=str, default=None, help="Model to load")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CELESTE RL - Deep Q-Network Training")
    print("=" * 60)
    
    # Create environment
    env = CelesteEnv(room=args.room, max_steps=args.max_steps, use_simple_actions=True)
    
    state_dim = env._get_obs_dim()
    action_dim = env.n_actions
    
    print(f"\nEnvironment:")
    print(f"  Room:       {args.room}")
    print(f"  State dim:  {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:     {device}")
    
    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        batch_size=128,
        buffer_size=200000,
        device=device
    )
    
    # Load model if specified
    if args.model:
        agent.load(args.model)
        print(f"  Loaded model from {args.model}")
    
    if args.eval_only:
        evaluate(env, agent, num_episodes=20)
    else:
        # Train
        rewards, heights = train(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            log_interval=50,
            save_interval=200,
            save_dir="models"
        )
        
        # Save training data
        with open("docs/training_data.pkl", "wb") as f:
            pickle.dump({'rewards': rewards, 'heights': heights}, f)
        
        # Plot
        plot_results(rewards, heights)
        
        # Evaluate
        evaluate(env, agent, num_episodes=20)


if __name__ == "__main__":
    main()
