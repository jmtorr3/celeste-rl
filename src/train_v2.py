"""
Training V2 - With Exploration Bonuses
Adds bonus for visiting new positions to encourage finding alternate paths.

Usage:
    python scripts/train_v2.py
    python scripts/train_v2.py --episodes 5000
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import DQNAgent
import torch

# Import base PICO8 stuff
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pyleste'))
from PICO8 import PICO8
from Carts.Celeste import Celeste
import CelesteUtils as utils


class CelesteEnvV2:
    """
    Environment with exploration bonuses.
    Rewards visiting new (x, y) positions to encourage finding alternate paths.
    """
    
    SIMPLE_ACTIONS = [
        0, 1, 2, 16, 17, 18, 32, 33, 34, 36, 37, 38, 40, 41, 42
    ]
    
    def __init__(self, room=0, max_steps=500):
        self.room = room
        self.max_steps = max_steps
        self.actions = self.SIMPLE_ACTIONS
        self.n_actions = len(self.actions)
        
        self.p8 = None
        self.step_count = 0
        
        # Tracking
        self.visited_positions = set()  # Track ALL positions visited this episode
        self.best_height = float('inf')
        self.no_progress_steps = 0
        
        # Global visit counts (persists across episodes for curiosity)
        self.global_visit_counts = {}
    
    @property
    def action_space(self):
        class ActionSpace:
            def __init__(self, n):
                self.n = n
            def sample(self):
                return np.random.randint(self.n)
        return ActionSpace(self.n_actions)
    
    def _get_player(self):
        for obj in self.p8.game.objects:
            if type(obj).__name__ == 'player':
                return obj
        return None
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.p8 = PICO8(Celeste)
        utils.load_room(self.p8, self.room)
        utils.skip_player_spawn(self.p8)
        
        self.step_count = 0
        self.visited_positions = set()
        self.no_progress_steps = 0
        
        player = self._get_player()
        self.best_height = player.y if player else 128
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.p8.set_btn_state(self.actions[action])
        self.p8.step()
        self.step_count += 1
        
        player = self._get_player()
        
        # === REWARD CALCULATION ===
        reward = 0.0
        terminated = False
        
        if player is None:
            # Death
            reward = -10.0
            terminated = True
        elif player.y < -8:
            # Level complete!
            reward = 300.0
            terminated = True
        else:
            # Discretize position for visit tracking
            pos = (int(player.x // 4), int(player.y // 4))  # 4-pixel grid
            
            # === EXPLORATION BONUS ===
            if pos not in self.visited_positions:
                self.visited_positions.add(pos)
                reward += 0.5  # Bonus for new position THIS episode
                
                # Extra bonus for globally rare positions
                self.global_visit_counts[pos] = self.global_visit_counts.get(pos, 0) + 1
                if self.global_visit_counts[pos] < 10:
                    reward += 0.2  # Extra bonus for rarely-visited spots
            
            # === HEIGHT BONUS ===
            if player.y < self.best_height:
                height_gained = self.best_height - player.y
                reward += height_gained * 3.0  # Strong height bonus
                self.best_height = player.y
                self.no_progress_steps = 0
            else:
                self.no_progress_steps += 1
            
            # === STUCK PENALTY ===
            if self.no_progress_steps > 100:
                reward -= 0.2
            
            # === POSITION-BASED HINTS ===
            # Give small bonus for being on the LEFT side when high up
            # (because the exit is on the left)
            if player.y < 50 and player.x < 60:
                reward += 0.1  # Encourage left side when near top
            
            # Small time penalty
            reward -= 0.01
            
            # Early termination if stuck too long
            if self.no_progress_steps > 150:
                terminated = True
        
        truncated = self.step_count >= self.max_steps
        
        info = {
            'player_x': player.x if player else 0,
            'player_y': player.y if player else 0,
            'max_height': self.best_height,
            'visited_count': len(self.visited_positions),
            'player_alive': player is not None,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        player = self._get_player()
        if player is None:
            return np.zeros(6, dtype=np.float32)
        
        return np.array([
            player.x / 64 - 1,
            player.y / 64 - 1,
            player.spd.x / 4,
            player.spd.y / 4,
            player.grace / 6,
            float(player.djump),
        ], dtype=np.float32)
    
    def render(self):
        return str(self.p8.game)
    
    def get_action_meaning(self, action):
        btn = self.actions[action]
        parts = []
        if btn & 1: parts.append("L")
        if btn & 2: parts.append("R")
        if btn & 4: parts.append("U")
        if btn & 8: parts.append("D")
        if btn & 16: parts.append("J")
        if btn & 32: parts.append("X")
        return "+".join(parts) if parts else "."


def train(num_episodes=3000):
    env = CelesteEnvV2(room=0, max_steps=500)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    agent = DQNAgent(
        state_dim=6,
        action_dim=env.n_actions,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,  # Keep more exploration
        epsilon_decay=0.9995,
        batch_size=128,
        buffer_size=200000,
        device=device
    )
    
    Path("models").mkdir(exist_ok=True)
    
    rewards = []
    heights = []
    completions = 0
    best_height = float('inf')
    
    print("=" * 60)
    print("TRAINING V2 - With Exploration Bonuses")
    print("=" * 60)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
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
        
        if info['max_height'] < -8:
            completions += 1
        
        if info['max_height'] < best_height:
            best_height = info['max_height']
            agent.save("models/model_v2_best.pt")
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards[-50:])
            avg_height = np.mean(heights[-50:])
            recent_complete = sum(1 for h in heights[-50:] if h < -8)
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward:     {avg_reward:.1f}")
            print(f"  Avg Height:     {avg_height:.1f}")
            print(f"  Best Height:    {best_height:.1f}")
            print(f"  Completions:    {recent_complete}/50 recent, {completions} total")
            print(f"  Epsilon:        {agent.epsilon:.3f}")
            print(f"  Unique pos:     {len(env.global_visit_counts)} discovered")
    
    agent.save("models/model_v2_final.pt")
    
    # Save data
    Path("docs").mkdir(exist_ok=True)
    with open("docs/training_v2.pkl", "wb") as f:
        pickle.dump({'rewards': rewards, 'heights': heights, 'completions': completions}, f)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Total completions: {completions}/{num_episodes}")
    print(f"Best height: {best_height}")
    print(f"{'='*60}")
    
    # Evaluate
    print("\nEvaluating...")
    agent.epsilon = 0.05
    
    eval_completions = 0
    for ep in range(50):
        state, _ = env.reset()
        while True:
            action = agent.select_action(state, training=True)
            state, _, done, trunc, info = env.step(action)
            if done or trunc:
                break
        if info['max_height'] < -8:
            eval_completions += 1
            print(f"  Eval {ep+1}: ✓ COMPLETE")
        else:
            print(f"  Eval {ep+1}: height={info['max_height']:.0f}")
    
    print(f"\nEval Success Rate: {eval_completions}/50 ({100*eval_completions/50:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    args = parser.parse_args()
    
    train(args.episodes)
