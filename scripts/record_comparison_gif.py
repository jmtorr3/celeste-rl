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


def _classify_outcome(info):
    if info.get('completed', False):
        return ('COMPLETE', COLOR_COMPLETE)
    if not info.get('player_alive', True):
        return (f"died (h={info.get('max_height', 0):.0f})", COLOR_DIED)
    return (f"timeout (h={info.get('max_height', 0):.0f})", COLOR_TIMEOUT)


def run_episodes(model_path, epsilon, n_episodes, room=0, max_steps=500):
    """Run N episodes back-to-back, return list of (frames, outcome) per episode."""
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

    episodes = []
    for _ in range(n_episodes):
        frames = []
        state, _ = env.reset()
        info = {}
        while True:
            action = agent.select_action(state, training=(epsilon > 0))
            state, _, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        episodes.append((frames, _classify_outcome(info)))
    return episodes


def run_random_episodes(n_episodes, room=0, max_steps=500):
    """Random-action baseline — N episodes."""
    env = CelesteEnv(room=room, max_steps=max_steps)
    episodes = []
    for _ in range(n_episodes):
        frames = []
        state, _ = env.reset()
        info = {}
        while True:
            state, _, terminated, truncated, info = env.step(env.action_space.sample())
            frames.append(env.render())
            if terminated or truncated:
                break
        episodes.append((frames, _classify_outcome(info)))
    return episodes


def render_panel(label, ascii_text, ep_idx, total_eps, outcome, tally, font_grid, font_label):
    """
    One panel:
      header  = method label + running tally (e.g. "dqn_r1     3/7 complete")
      body    = live ASCII grid
      footer  = current episode marker + outcome (color-coded if episode ended)

    tally  = (completes_so_far, episodes_finished_so_far)
    outcome = (text, color) if current episode finished, else None (still running)
    """
    img = Image.new("RGB", (PANEL_W, PANEL_H), PANEL_BG)
    draw = ImageDraw.Draw(img)

    # Header — method name + running tally
    completes, finished = tally
    tally_str = f"{completes}/{finished} complete" if finished else ""
    draw.text((10, 4), label, fill=LABEL_FG, font=font_label)
    if tally_str:
        # Right-align the tally
        bbox = draw.textbbox((0, 0), tally_str, font=font_label)
        tw = bbox[2] - bbox[0]
        draw.text((PANEL_W - tw - 10, 4), tally_str, fill=COLOR_RUNNING, font=font_label)

    # Body — ASCII grid
    lines = ascii_text.split("\n")
    y = HEADER_H
    for line in lines[:18]:
        draw.text((10, y), line, fill=PANEL_FG, font=font_grid)
        y += CHAR_H

    # Footer — episode marker + outcome
    footer_y = PANEL_H - FOOTER_H + 4
    if outcome is not None:
        text, color = outcome
        draw.text((10, footer_y), f"ep {ep_idx}/{total_eps}  {text}", fill=color, font=font_label)
    else:
        draw.text((10, footer_y), f"ep {ep_idx}/{total_eps}  running…", fill=COLOR_RUNNING, font=font_label)

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


def _frame_position(global_step, episodes):
    """
    Given a global step index and a list of (frames, outcome) episodes for one method,
    return (ascii_to_show, ep_idx, total_eps, outcome_or_None, completes_so_far,
    episodes_finished_so_far).

    Episodes are concatenated end-to-end. After the last episode finishes, the panel
    freezes on the final frame.
    """
    total_eps = len(episodes)
    completes_so_far = 0
    cursor = 0
    for ep_idx, (frames, outcome) in enumerate(episodes, start=1):
        if global_step < cursor + len(frames):
            local = global_step - cursor
            is_final_step_of_ep = (local == len(frames) - 1)
            return (
                frames[local],
                ep_idx,
                total_eps,
                outcome if is_final_step_of_ep else None,
                completes_so_far,
                ep_idx - 1,
            )
        cursor += len(frames)
        if outcome[0] == 'COMPLETE':
            completes_so_far += 1

    # Past the last episode: freeze on its final frame
    last_frames, last_outcome = episodes[-1]
    return (
        last_frames[-1],
        total_eps,
        total_eps,
        last_outcome,
        completes_so_far,
        total_eps,
    )


def build_comparison_gif(runs, output, epsilon=0.05, fps=15, room=0, max_steps=500,
                         include_random=True, cols=None, n_episodes=10):
    """Run N episodes per method, sync them, write a single comparison GIF."""
    font_grid = load_font(13)
    font_label = load_font(13)

    method_data = []  # list of (label, episodes) where episodes = [(frames, outcome), ...]

    if include_random:
        print(f"  random ({n_episodes} eps)... ", end="", flush=True)
        episodes = run_random_episodes(n_episodes, room, max_steps)
        completes = sum(1 for _, o in episodes if o[0] == 'COMPLETE')
        method_data.append(("random", episodes))
        print(f"{completes}/{n_episodes} complete")

    for run_id, path in runs.items():
        print(f"  {run_id} ({n_episodes} eps)... ", end="", flush=True)
        try:
            episodes = run_episodes(path, epsilon, n_episodes, room, max_steps)
            completes = sum(1 for _, o in episodes if o[0] == 'COMPLETE')
            method_data.append((run_id, episodes))
            print(f"{completes}/{n_episodes} complete")
        except Exception as e:
            print(f"ERROR: {e}")

    if not method_data:
        print("No methods ran successfully — nothing to record.")
        return

    # Sort: best methods (most completes) first
    method_data.sort(key=lambda m: -sum(1 for _, o in m[1] if o[0] == 'COMPLETE'))

    # Total length per method = sum of all its episode frame counts.
    # The GIF runs until the longest method finishes.
    method_total_frames = [sum(len(f) for f, _ in eps) for _, eps in method_data]
    timeline_length = max(method_total_frames)
    print(f"\nTimeline: {timeline_length} frames "
          f"(longest method ran {n_episodes} episodes summing to {timeline_length} frames)")

    # Grid layout
    n = len(method_data)
    if cols is None:
        cols = 3 if n >= 4 else (2 if n >= 2 else 1)

    # Build composite frames
    composite_frames = []
    for step in range(timeline_length):
        panels = []
        for label, episodes in method_data:
            ascii_grid, ep_idx, total_eps, outcome, completes, finished = _frame_position(
                step, episodes,
            )
            panels.append(render_panel(
                label, ascii_grid, ep_idx, total_eps,
                outcome, (completes, finished),
                font_grid, font_label,
            ))
        composite_frames.append(composite_grid(panels, cols))

    # Hold the final frame for ~3 seconds so viewers can read the final tallies
    final_panels = []
    for label, episodes in method_data:
        last_frames, last_outcome = episodes[-1]
        completes = sum(1 for _, o in episodes if o[0] == 'COMPLETE')
        final_panels.append(render_panel(
            label, last_frames[-1], len(episodes), len(episodes),
            last_outcome, (completes, len(episodes)),
            font_grid, font_label,
        ))
    final = composite_grid(final_panels, cols)
    for _ in range(int(fps * 3)):
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
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per method (default: 10)')
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

    print(f"Recording {args.episodes} episodes per method at ε={args.epsilon}:")
    build_comparison_gif(
        runs, args.output,
        epsilon=args.epsilon,
        fps=args.fps,
        room=args.room,
        max_steps=args.max_steps,
        include_random=args.include_random,
        cols=args.cols,
        n_episodes=args.episodes,
    )


if __name__ == '__main__':
    main()
