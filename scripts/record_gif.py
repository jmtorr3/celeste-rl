"""
Record a GIF of a trained agent successfully completing the level.

Renders each frame of `env.render()` (the ASCII grid) as a terminal-style
PIL image, then stitches them into an animated GIF. Retries until a
completion run is captured.

Usage:
    python scripts/record_gif.py --run-id v3_r9
    python scripts/record_gif.py --model runs/v3_r9/checkpoint_ep1950.pt --output demo.gif
    python scripts/record_gif.py --run-id v3_r9 --epsilon 0.05 --fps 20 --max-attempts 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

from src.environment import CelesteEnv
from src.agent import DQNAgent
from src.network import DuelingDQN


# Terminal-style colors (dark mode)
BG = (16, 24, 32)
FG = (220, 220, 220)
ACCENT = (180, 240, 180)  # for status footer

# Try a few fonts in order — first one that exists wins
FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",                    # macOS
    "/System/Library/Fonts/Monaco.ttf",                   # macOS fallback
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",         # Linux variant
    "/Library/Fonts/Courier New.ttf",                     # macOS legacy
]


def load_font(size=14):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_frame(env, font, episode, step, total_completions, status=""):
    """Render the env's ASCII grid as a PIL image with a status footer."""
    text = env.render()
    lines = text.split("\n")

    # Each line is the ASCII map row (32 chars wide, 16 rows + a few status lines)
    # Use a fixed canvas size so frames are consistent
    char_w, char_h = 9, 16
    grid_w = max(len(line) for line in lines) * char_w + 20
    grid_h = len(lines) * char_h + 60  # extra space for footer

    img = Image.new("RGB", (grid_w, grid_h), BG)
    draw = ImageDraw.Draw(img)

    for i, line in enumerate(lines):
        draw.text((10, 10 + i * char_h), line, fill=FG, font=font)

    # Footer
    footer_y = 10 + len(lines) * char_h + 8
    draw.text((10, footer_y),
              f"v3_r9  ep{episode}  step{step:>3}  completions: {total_completions}",
              fill=ACCENT, font=font)
    if status:
        draw.text((10, footer_y + char_h + 4), status, fill=ACCENT, font=font)

    return img


def record_run(model_path, output, epsilon=0.05, max_attempts=30, fps=20,
               room=0, max_steps=500, dueling=True):
    env = CelesteEnv(room=room, max_steps=max_steps)
    agent = DQNAgent(
        state_dim=env._get_obs_dim(),
        action_dim=env.n_actions,
        device="cpu",
        network_cls=DuelingDQN if dueling else None,
    )
    agent.load(model_path)
    agent.epsilon = epsilon
    print(f"Loaded {model_path} (epsilon={epsilon})")

    font = load_font(size=14)

    for attempt in range(1, max_attempts + 1):
        frames = []
        state, _ = env.reset()
        step = 0

        while True:
            action = agent.select_action(state, training=(epsilon > 0))
            state, _, terminated, truncated, info = env.step(action)
            step += 1

            frames.append(render_frame(env, font, attempt, step, attempt - 1))

            if terminated or truncated:
                break

        completed = info.get("completed", False)
        max_h = info.get("max_height", 999)
        outcome = "COMPLETE" if completed else (
            f"died h={max_h:.0f}" if not info.get("player_alive") else f"timeout h={max_h:.0f}"
        )
        print(f"  Attempt {attempt:>2}: {step} frames, {outcome}")

        if completed:
            # Hold the final frame for ~1s so the loop has a clear ending
            final = render_frame(env, font, attempt, step, attempt, status="LEVEL COMPLETE")
            for _ in range(int(fps)):
                frames.append(final)

            save_gif(frames, output, fps=fps)
            print(f"\nSuccess on attempt {attempt}.")
            print(f"GIF saved to {output} ({len(frames)} frames @ {fps}fps)")
            return

    print(f"\nNo successful run in {max_attempts} attempts. Try a different checkpoint or raise --max-attempts.")


def save_gif(frames, output, fps=20):
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def resolve_model_path(args):
    if args.model:
        return args.model
    if args.run_id:
        return os.path.join("runs", args.run_id, "best.pt")
    raise SystemExit("Must provide --run-id or --model")


def resolve_output_path(args):
    if args.output:
        return args.output
    if args.run_id:
        run_dir = os.path.join("runs", args.run_id)
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, f"{args.run_id}_demo.gif")
    return "demo.gif"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a GIF of a successful agent run")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID — resolves to runs/{run_id}/best.pt")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit model path (overrides --run-id)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GIF path (default: runs/{run_id}/{run_id}_demo.gif)")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Exploration rate during recording (default: 0.05)")
    parser.add_argument("--fps", type=int, default=20,
                        help="GIF frame rate (default: 20)")
    parser.add_argument("--max-attempts", type=int, default=30,
                        help="Max episodes to try before giving up (default: 30)")
    parser.add_argument("--room", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--no-dueling", action="store_true",
                        help="Use plain DQN architecture (default is DuelingDQN)")
    args = parser.parse_args()

    model_path = resolve_model_path(args)
    output_path = resolve_output_path(args)
    record_run(
        model_path,
        output=output_path,
        epsilon=args.epsilon,
        max_attempts=args.max_attempts,
        fps=args.fps,
        room=args.room,
        max_steps=args.max_steps,
        dueling=not args.no_dueling,
    )
