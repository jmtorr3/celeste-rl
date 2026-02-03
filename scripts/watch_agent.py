"""
Watch a trained agent play Celeste Classic.

Usage:
    python scripts/watch_agent.py
    python scripts/watch_agent.py --model models/dqn_best.pt --delay 0.05
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent


def watch_agent(
    model_path: str = "models/dqn_best.pt",
    room: int = 0,
    num_episodes: int = 3,
    delay: float = 0.03
):
    """Watch a trained agent play."""
    
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
    
    print(f"\nWatching agent play room {room}...")
    print("Press Ctrl+C to stop\n")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"\n{'='*50}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*50}\n")
        
        try:
            while True:
                print("\033[H\033[J", end="")  # Clear screen
                
                action = agent.select_action(state, training=False)
                action_name = env.get_action_meaning(action)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                print(env.render())
                print()
                print(f"Episode: {episode + 1}/{num_episodes} | Step: {step}")
                print(f"Action: {action_name}")
                print(f"Position: ({info['player_x']}, {info['player_y']})")
                print(f"Reward: {reward:+.2f} | Total: {episode_reward:.2f}")
                
                state = next_state
                
                if terminated or truncated:
                    print("\n" + "-" * 50)
                    if info['player_alive'] and info.get('player_y', 999) < -8:
                        print("🎉 LEVEL COMPLETE!")
                    elif not info['player_alive']:
                        print("💀 DIED")
                    else:
                        print("⏱️ TIME LIMIT")
                    print(f"Final reward: {episode_reward:.2f}")
                    time.sleep(1)
                    break
                
                time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            return
    
    print("\nFinished watching!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained agent")
    parser.add_argument("--model", type=str, default="models/dqn_best.pt")
    parser.add_argument("--room", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.03)
    
    args = parser.parse_args()
    
    watch_agent(
        model_path=args.model,
        room=args.room,
        num_episodes=args.episodes,
        delay=args.delay
    )
