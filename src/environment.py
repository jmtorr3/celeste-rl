"""
Celeste Classic RL Environment
A Gymnasium-compatible wrapper for Pyleste
"""

import numpy as np
from typing import Optional, Tuple, List
import sys
import os

# Add pyleste to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyleste'))

from PICO8 import PICO8
from Carts.Celeste import Celeste
import CelesteUtils as utils


class CelesteEnv:
    """
    A Gymnasium-style environment for Celeste Classic.
    
    Player attributes (from Pyleste):
        x, y: Position (int)
        spd.x, spd.y: Speed (float, via Vector object)
        rem.x, rem.y: Subpixel remainder (float, via Vector object)
        grace: Coyote time frames (int, 0-6)
        djump: Dashes remaining (int, 0 or 1)
        dash_time: Frames remaining in dash (int)
    """
    
    # Common useful actions (reduced action space)
    SIMPLE_ACTIONS = [
        0,   # nothing
        1,   # left
        2,   # right
        16,  # jump
        17,  # jump + left
        18,  # jump + right
        32,  # dash
        33,  # dash + left
        34,  # dash + right
        36,  # dash + up
        37,  # dash + up + left
        38,  # dash + up + right
        40,  # dash + down
        41,  # dash + down + left
        42,  # dash + down + right
    ]
    
    ALL_ACTIONS = list(range(64))
    
    def __init__(
        self,
        room: int = 0,
        max_steps: int = 1000,
        use_simple_actions: bool = True,
        custom_actions: Optional[List[int]] = None,
    ):
        """
        Initialize the Celeste environment.
        
        Args:
            room: Level number (0-30). 0 is 100m, 20 is 2100m, 30 is summit.
            max_steps: Maximum steps before episode is truncated.
            use_simple_actions: If True, use reduced action space (15 actions).
            custom_actions: Optional list of action integers to use instead.
        """
        self.room = room
        self.max_steps = max_steps
        
        if custom_actions is not None:
            self.actions = custom_actions
        elif use_simple_actions:
            self.actions = self.SIMPLE_ACTIONS
        else:
            self.actions = self.ALL_ACTIONS
        
        self.n_actions = len(self.actions)
        
        self.p8 = None
        self.step_count = 0
        
        # For tracking
        self.episode_reward = 0
        self.max_height_reached = float('inf')
        self.prev_y = None
        self.prev_x = None
        self.stuck_count = 0
        self.best_height_this_episode = float('inf')
        
    @property
    def action_space(self):
        """Simple action space with sample() method."""
        class ActionSpace:
            def __init__(self, n):
                self.n = n
            def sample(self):
                return np.random.randint(self.n)
        return ActionSpace(self.n_actions)
    
    @property
    def observation_space(self):
        """Observation space shape."""
        return (self._get_obs_dim(),)
    
    def _get_obs_dim(self) -> int:
        """Get observation dimension."""
        return 31
    
    def _get_player(self):
        """Get the active player object (not player_spawn)."""
        for obj in self.p8.game.objects:
            if type(obj).__name__ == 'player':
                return obj
        return None

    def _is_room_transition(self) -> bool:
        """
        True when Pyleste has loaded the next room after the player exited.
        player_spawn presence means a room was just initialised — the player
        object exists in the new room, not in ours, so _get_player() returns
        None. Treat this as level complete, NOT as death.
        """
        for obj in self.p8.game.objects:
            if type(obj).__name__ == 'player_spawn':
                return True
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.p8 = PICO8(Celeste)
        utils.load_room(self.p8, self.room)
        utils.skip_player_spawn(self.p8)
        
        self.step_count = 0
        self.episode_reward = 0
        self.stuck_count = 0
        self.milestones_hit = set()

        player = self._get_player()
        self.prev_y = player.y if player else 128
        self.prev_x = player.x if player else 0
        self.max_height_reached = self.prev_y
        self.best_height_this_episode = self.prev_y
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int):
        """Take a step in the environment."""
        btn_state = self.actions[action]
        
        self.p8.set_btn_state(btn_state)
        self.p8.step()
        self.step_count += 1
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        
        self.episode_reward += reward
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        player = self._get_player()
        
        if player is None:
            return np.zeros(self._get_obs_dim(), dtype=np.float32)
        
        tile_x = int(player.x / 8)
        tile_y = int(player.y / 8)
        tile_grid = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                tx = max(0, min(15, tile_x + dx))
                ty = max(0, min(15, tile_y + dy))
                tile_grid.append(float(self.p8.game.tile_at(tx, ty)))

        obs = np.array([
            player.x / 128.0,
            player.y / 128.0,
            player.spd.x / 4,
            player.spd.y / 4,
            player.grace / 6,
            float(player.djump),
            *tile_grid,
        ], dtype=np.float32)

        return obs
    
    def _compute_reward(self) -> float:
        """
        Compute reward with improved shaping:
        - Big bonus for reaching new heights
        - Small bonus for moving (exploration)
        - Penalty for death and being stuck
        """
        player = self._get_player()
        
        if player is None:
            if self._is_room_transition():
                return 500.0  # Room exit = level complete
            return -5.0  # Actual death

        if player.y < -8:
            return 500.0  # Level complete bonus (fallback — should be caught above)
        
        reward = 0.0
        
        # New height bonus — progressive multiplier, stronger near the exit
        if player.y < self.best_height_this_episode:
            height_gained = self.best_height_this_episode - player.y
            progress_scale = max(1.0, (96 - player.y) / 24.0)  # 1x at bottom, 4x near exit
            reward += height_gained * progress_scale
            self.best_height_this_episode = player.y

        # Milestone bonuses — first time reaching each checkpoint per episode
        for threshold, bonus in ((40, 20.0), (20, 40.0), (10, 80.0), (0, 150.0), (-5, 300.0)):
            if player.y < threshold and threshold not in self.milestones_hit:
                self.milestones_hit.add(threshold)
                reward += bonus
        
        # Movement bonus (encourage exploration)
        dx = abs(player.x - self.prev_x)
        dy = abs(player.y - self.prev_y)
        movement = dx + dy
        
        if movement > 0:
            reward += 0.01
            self.stuck_count = 0
        else:
            self.stuck_count += 1
            if self.stuck_count > 30:
                reward -= 0.1
        
        # Time penalty
        reward -= 0.01
        
        # Update previous position
        self.prev_x = player.x
        self.prev_y = player.y
        
        if player.y < self.max_height_reached:
            self.max_height_reached = player.y
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if episode is terminated."""
        player = self._get_player()

        if player is None:
            return True  # either room transition (complete) or death

        if player.y < -8:
            return True

        if self.stuck_count > 150:
            return True

        return False

    def _is_complete(self) -> bool:
        """True if the episode ended in a level completion (not death)."""
        if self._get_player() is None and self._is_room_transition():
            return True
        if self._get_player() is not None and self._get_player().y < -8:
            return True
        return False
    
    def _get_info(self) -> dict:
        """Get additional info about current state."""
        player = self._get_player()
        
        if player is None:
            return {
                "step": self.step_count,
                "player_alive": False,
                "player_x": 0,
                "player_y": 0,
                "player_spd_x": 0,
                "player_spd_y": 0,
                "max_height": self.max_height_reached,
                "episode_reward": self.episode_reward,
                "completed": self._is_complete(),
            }

        return {
            "step": self.step_count,
            "player_alive": True,
            "player_x": player.x,
            "player_y": player.y,
            "player_spd_x": player.spd.x,
            "player_spd_y": player.spd.y,
            "grace": player.grace,
            "djump": player.djump,
            "dash_time": player.dash_time,
            "max_height": self.max_height_reached,
            "episode_reward": self.episode_reward,
            "completed": False,  # only set on termination step
        }
    
    def render(self, mode: str = "text") -> Optional[str]:
        """Render the current state."""
        if mode == "text":
            return str(self.p8.game)
        return None
    
    def get_action_meaning(self, action: int) -> str:
        """Get human-readable meaning of an action."""
        btn_state = self.actions[action]
        parts = []
        if btn_state & 1:
            parts.append("left")
        if btn_state & 2:
            parts.append("right")
        if btn_state & 4:
            parts.append("up")
        if btn_state & 8:
            parts.append("down")
        if btn_state & 16:
            parts.append("jump")
        if btn_state & 32:
            parts.append("dash")
        return " + ".join(parts) if parts else "nothing"
    
    def close(self):
        """Clean up resources."""
        self.p8 = None


if __name__ == "__main__":
    print("Testing CelesteEnv...")
    env = CelesteEnv(room=0, max_steps=500)
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Number of actions: {env.n_actions}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"Episode ended at step {step}, reward: {total_reward:.2f}")
            break
    
    print("✅ Environment working!")
