"""
Record one episode of every model in runs/ and stitch them into a single
comparison GIF: each method gets its own labeled panel running in lockstep.

Layout: auto-detects rows/cols based on number of models. Each panel shows
the run id, the live ASCII grid for that agent, and the final outcome
(COMPLETE / died / timeout) once the episode ends. Panels for fast-failing
methods freeze on their final frame while longer-running ones continue.

Usage:
    python scripts/record_comparison_gif.py
    python scripts/record_comparison_gif.py --epsilon 0.05 --fps 15
    python scripts/record_comparison_gif.py --include-random --output docs/comparison.gif

Outputs:
    docs/comparison.gif    side-by-side animation (default)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent
from src.network import DuelingDQN

from scripts.evaluate import detect_architecture
from scripts.compare_all import discover_runs


# Panel rendering
PANEL_BG = (16, 24, 32)
PANEL_FG = (220, 220, 220)
LABEL_FG = (255, 255, 255)
COLOR_COMPLETE = (90, 220, 120)
COLOR_DIED = (240, 100, 100)
COLOR_TIMEOUT = (200, 200, 200)
COLOR_RUNNING = (180, 180, 180)

CHAR_W, CHAR_H = 8, 14
PANEL_W = 32 * CHAR_W + 20    # ASCII grid is 32 chars wide + padding
HEADER_H = 22
FOOTER_H = 22
PANEL_H = 18 * CHAR_H + HEADER_H + FOOTER_H + 10


def load_font(size):
    for path in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def run_episode(model_path, epsilon, room=0, max_steps=500):
    """Run one episode, capturing each frame's ASCII grid + final outcome."""
    env = CelesteEnv(room=room, max_steps=max_steps)
    arch = detect_architecture(model_path)
    agent = DQNAgent(
        state_dim=env._get_obs_dim(),
        action_dim=env.n_actions,
        device='cpu',
        network_cls=DuelingDQN if arch == 'dueling' else None,
    )
    agent.load(model_path)
    agent.epsilon = epsilon

    frames = []
    state, _ = env.reset()
    info = {}
    while True:
        action = agent.select_action(state, training=(epsilon > 0))
        state, _, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    if info.get('completed', False):
        outcome = ('COMPLETE', COLOR_COMPLETE)
    elif not info.get('player_alive', True):
        outcome = (f"died (h={info.get('max_height', 0):.0f})", COLOR_DIED)
    else:
        outcome = (f"timeout (h={info.get('max_height', 0):.0f})", COLOR_TIMEOUT)

    return frames, outcome


def run_random_episode(room=0, max_steps=500):
    """Random-action baseline episode for comparison."""
    env = CelesteEnv(room=room, max_steps=max_steps)
    frames = []
    state, _ = env.reset()
    info = {}
    while True:
        state, _, terminated, truncated, info = env.step(env.action_space.sample())
        frames.append(env.render())
        if terminated or truncated:
            break

    if info.get('completed', False):
        outcome = ('COMPLETE', COLOR_COMPLETE)
    elif not info.get('player_alive', True):
        outcome = (f"died (h={info.get('max_height', 0):.0f})", COLOR_DIED)
    else:
        outcome = (f"timeout (h={info.get('max_height', 0):.0f})", COLOR_TIMEOUT)
    return frames, outcome


def render_panel(label, ascii_text, step, total_steps, outcome, font_grid, font_label):
    """One panel: header (label), body (ascii), footer (step / outcome)."""
    img = Image.new("RGB", (PANEL_W, PANEL_H), PANEL_BG)
    draw = ImageDraw.Draw(img)

    # Header — method name
    draw.text((10, 4), label, fill=LABEL_FG, font=font_label)

    # Body — ASCII grid
    lines = ascii_text.split("\n")
    y = HEADER_H
    for line in lines[:18]:
        draw.text((10, y), line, fill=PANEL_FG, font=font_grid)
        y += CHAR_H

    # Footer — step counter and outcome (color-coded once episode ends)
    footer_y = PANEL_H - FOOTER_H + 4
    if outcome is not None:
        text, color = outcome
        draw.text((10, footer_y), f"step {step}/{total_steps}  {text}", fill=color, font=font_label)
    else:
        draw.text((10, footer_y), f"step {step}/{total_steps}  …", fill=COLOR_RUNNING, font=font_label)

    return img


def composite_grid(panel_imgs, cols):
    """Combine panel images into a grid layout."""
    rows = math.ceil(len(panel_imgs) / cols)
    w = PANEL_W * cols
    h = PANEL_H * rows
    canvas = Image.new("RGB", (w, h), (8, 12, 16))
    for i, p in enumerate(panel_imgs):
        r, c = divmod(i, cols)
        canvas.paste(p, (c * PANEL_W, r * PANEL_H))
    return canvas


def build_comparison_gif(runs, output, epsilon=0.05, fps=15, room=0, max_steps=500,
                         include_random=True, cols=None):
    """Run one episode per method, sync them, write a comparison GIF."""
    font_grid = load_font(13)
    font_label = load_font(13)

    # Run each method once
    method_data = []  # list of (label, frames, outcome)

    if include_random:
        print(f"  random... ", end="", flush=True)
        frames, outcome = run_random_episode(room, max_steps)
        method_data.append(("random", frames, outcome))
        print(f"{len(frames)} frames, {outcome[0]}")

    for run_id, path in runs.items():
        print(f"  {run_id}... ", end="", flush=True)
        try:
            frames, outcome = run_episode(path, epsilon, room, max_steps)
            method_data.append((run_id, frames, outcome))
            print(f"{len(frames)} frames, {outcome[0]}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not method_data:
        print("No methods ran successfully — nothing to record.")
        return

    # Sort: best result first (COMPLETE > timeout > died)
    def sort_key(m):
        text, _ = m[2]
        if text == 'COMPLETE': return 0
        if text.startswith('timeout'): return 1
        return 2
    method_data.sort(key=sort_key)

    # Synchronize length: pad shorter episodes with their final frame
    total_steps = max(len(f) for _, f, _ in method_data)
    print(f"\nSyncing all panels to {total_steps} frames...")

    # Decide grid layout
    n = len(method_data)
    if cols is None:
        cols = 3 if n >= 4 else (2 if n >= 2 else 1)

    # Build composite frames
    composite_frames = []
    for step in range(total_steps):
        panels = []
        for label, frames, outcome in method_data:
            if step < len(frames):
                ascii_grid = frames[step]
                panel_outcome = None
            else:
                # Episode over — freeze on final frame, show outcome
                ascii_grid = frames[-1]
                panel_outcome = outcome
            panels.append(render_panel(
                label, ascii_grid, min(step + 1, len(frames)), len(frames),
                panel_outcome, font_grid, font_label,
            ))
        composite_frames.append(composite_grid(panels, cols))

    # Hold the final frame for ~2 seconds so viewers can read the outcomes
    final_panels = []
    for label, frames, outcome in method_data:
        final_panels.append(render_panel(
            label, frames[-1], len(frames), len(frames),
            outcome, font_grid, font_label,
        ))
    final = composite_grid(final_panels, cols)
    for _ in range(int(fps * 2)):
        composite_frames.append(final)

    # Write GIF
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(1000 / fps))
    composite_frames[0].save(
        output,
        save_all=True,
        append_images=composite_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"\nSaved comparison GIF to {output}")
    print(f"  {len(method_data)} panels  |  {len(composite_frames)} frames  |  {fps} fps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', type=str, default='runs')
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--cols', type=int, default=None,
                        help='Grid columns (default: auto)')
    parser.add_argument('--no-random', dest='include_random', action='store_false', default=True)
    parser.add_argument('--output', type=str, default='docs/comparison.gif')
    parser.add_argument('--override', action='append', default=[],
                        help='Per-run override: --override dqn_r1=runs/dqn_r1/checkpoint_ep5000.pt')
    args = parser.parse_args()

    # Reuse override parser from compare_all
    overrides = {}
    for s in args.override:
        if '=' in s:
            k, v = s.split('=', 1)
            overrides[k.strip()] = v.strip()

    runs = discover_runs(args.runs_dir, overrides)
    if not runs:
        print(f"No runs found under {args.runs_dir}/")
        return

    print(f"Recording one episode each at ε={args.epsilon}:")
    build_comparison_gif(
        runs, args.output,
        epsilon=args.epsilon,
        fps=args.fps,
        room=args.room,
        max_steps=args.max_steps,
        include_random=args.include_random,
        cols=args.cols,
    )


if __name__ == '__main__':
    main()
