"""
Export trained agent's inputs to TAS format for playback in real PICO-8.

Usage:
    python scripts/export_tas.py --model models/model_v2_best.pt
    python scripts/export_tas.py --model models/model_v2_best.pt --attempts 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyleste"
    ),
)

from src.agent import DQNAgent
from PICO8 import PICO8
from Carts.Celeste import Celeste
import CelesteUtils as utils
import numpy as np


def get_player(p8):
    for obj in p8.game.objects:
        if type(obj).__name__ == "player":
            return obj
    return None


def btn_to_tas(btn_state):
    """Convert button state to TAS string format."""
    parts = []
    if btn_state & 1:
        parts.append("L")
    if btn_state & 2:
        parts.append("R")
    if btn_state & 4:
        parts.append("U")
    if btn_state & 8:
        parts.append("D")
    if btn_state & 16:
        parts.append("Z")  # Jump = Z in PICO-8
    if btn_state & 32:
        parts.append("X")  # Dash = X in PICO-8
    return ",".join(parts) if parts else ""


def record_attempt(agent, room=0, max_frames=1000, epsilon=0.0):
    """Record one attempt and return (inputs, success, final_height)."""

    SIMPLE_ACTIONS = [0, 1, 2, 16, 17, 18, 32, 33, 34, 36, 37, 38, 40, 41, 42]

    p8 = PICO8(Celeste)
    utils.load_room(p8, room)
    utils.skip_player_spawn(p8)

    inputs = []
    agent.epsilon = epsilon

    for frame in range(max_frames):
        player = get_player(p8)

        if player is None:
            return inputs, False, 999  # Died

        if player.y < -8:
            return inputs, True, player.y  # Success!

        # Get observation
        obs = np.array(
            [
                player.x / 64 - 1,
                player.y / 64 - 1,
                player.spd.x / 4,
                player.spd.y / 4,
                player.grace / 6,
                float(player.djump),
            ],
            dtype=np.float32,
        )

        # Get action
        action_idx = agent.select_action(obs, training=(epsilon > 0))
        btn_state = SIMPLE_ACTIONS[action_idx]

        inputs.append(btn_state)

        p8.set_btn_state(btn_state)
        p8.step()

    player = get_player(p8)
    final_height = player.y if player else 999
    return inputs, False, final_height


def export_tas(model_path, output_path="tas_output.tas", attempts=5, epsilon=0.05):
    """Try multiple attempts and export the best one."""

    agent = DQNAgent(state_dim=6, action_dim=15, device="cpu")
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    best_inputs = None
    best_height = float("inf")
    success = False

    print(f"\nRecording {attempts} attempts...")

    for i in range(attempts):
        inputs, completed, height = record_attempt(agent, epsilon=epsilon)

        status = "✓ COMPLETE!" if completed else f"height={height:.0f}"
        print(f"  Attempt {i+1}: {len(inputs)} frames, {status}")

        if completed:
            if best_inputs is None or len(inputs) < len(best_inputs):
                best_inputs = inputs
                best_height = height
                success = True
        elif height < best_height and not success:
            best_inputs = inputs
            best_height = height

    if best_inputs is None:
        print("No valid attempts recorded!")
        return

    # Write TAS file
    with open(output_path, "w") as f:
        for btn_state in best_inputs:
            f.write(btn_to_tas(btn_state) + "\n")

    print(f"\n{'='*50}")
    print(f"Exported to: {output_path}")
    print(f"Frames: {len(best_inputs)}")
    print(f"Success: {success}")
    print(f"Best height: {best_height}")
    print(f"{'='*50}")

    # Also save as Python list for reference
    py_path = output_path.replace(".tas", ".py")
    with open(py_path, "w") as f:
        f.write(f"# TAS inputs for Celeste Classic\n")
        f.write(f"# Frames: {len(best_inputs)}\n")
        f.write(f"# Success: {success}\n\n")
        f.write(f"inputs = {best_inputs}\n")

    print(f"Also saved as: {py_path}")

    # Print instructions
    print(f"""
{'='*50}
HOW TO PLAY IN PICO-8:
{'='*50}

1. Download UniversalClassicTas:
   https://github.com/CelesteClassic/UniversalClassicTas

2. Copy '{output_path}' to the TAS tool folder

3. Run the TAS tool with PICO-8

4. Watch your AI play!

Alternative: Use the realtime script (less reliable):
   python scripts/play_realtime.py --tas {py_path}
{'='*50}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export agent to TAS format")
    parser.add_argument("--model", type=str, default="models/model_v2_best.pt")
    parser.add_argument("--output", type=str, default="tas_output.tas")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Exploration during recording"
    )

    args = parser.parse_args()

    export_tas(args.model, args.output, args.attempts, args.epsilon)
