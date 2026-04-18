"""
Export TAS replay data as (state, action) pairs for Behavioral Cloning.

Replays each TAS file through Pyleste, records the 31-dim observation and
the corresponding action index (into CelesteEnv.SIMPLE_ACTIONS) at every frame.

Usage:
    python scripts/export_tas.py
    python scripts/export_tas.py --tas-dir TAS_data/any% --output data/tas_transitions.pkl
    python scripts/export_tas.py --room 0 --verbose
"""

import argparse
import pickle
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv

SIMPLE_ACTIONS = CelesteEnv.SIMPLE_ACTIONS


def parse_tas(path: str):
    """Parse a .tas file into a list of button-state integers."""
    text = open(path).read().strip()
    if text.startswith('[]'):
        text = text[2:]
    return [int(x) for x in text.split(',') if x.strip()]


def action_to_idx(btn: int):
    """Map a raw button bitmask to its SIMPLE_ACTIONS index. Returns None if not found."""
    try:
        return SIMPLE_ACTIONS.index(btn)
    except ValueError:
        return None


def export_room(room: int, inputs: list, verbose: bool = False):
    """
    Replay inputs through a single room, collect (obs, action_idx) pairs.
    Returns (transitions, skipped_count).
    """
    env = CelesteEnv(room=room, max_steps=len(inputs) + 100)
    obs, _ = env.reset()

    transitions = []
    skipped = 0

    for btn in inputs:
        action_idx = action_to_idx(btn)

        if action_idx is None:
            if verbose:
                print(f"  [room {room}] unrecognized input {btn}, skipping")
            skipped += 1
            env.step(0)
            obs = env._get_obs()
            continue

        transitions.append((obs.copy(), action_idx))
        obs, _, terminated, truncated, info = env.step(action_idx)

        if terminated:
            if info.get('player_y', 999) < -8 and verbose:
                print(f"  [room {room}] complete at step {len(transitions)}")
            break
        if truncated:
            break

    return transitions, skipped


def main():
    parser = argparse.ArgumentParser(description='Export TAS transitions for BC training')
    parser.add_argument('--tas-dir', type=str, default='TAS_data/any%')
    parser.add_argument('--output', type=str, default='data/tas_transitions.pkl')
    parser.add_argument('--room', type=int, default=None, help='Export single room only (0-indexed)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    tas_dir = Path(args.tas_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    if args.room is not None:
        tas_files = [(args.room, tas_dir / f'TAS{args.room + 1}.tas')]
    else:
        tas_files = sorted(
            [(int(p.stem.replace('TAS', '')) - 1, p) for p in tas_dir.glob('TAS*.tas')],
            key=lambda x: x[0]
        )

    all_transitions = []
    total_skipped = 0

    print(f"TAS dir:  {tas_dir}")
    print(f"Output:   {output_path}")
    print(f"Rooms:    {len(tas_files)}")
    print("=" * 55)

    for room, tas_path in tas_files:
        if not tas_path.exists():
            print(f"  Room {room:2d} | {tas_path.name} not found, skipping")
            continue

        inputs = parse_tas(str(tas_path))
        transitions, skipped = export_room(room, inputs, verbose=args.verbose)
        total_skipped += skipped
        all_transitions.extend(transitions)

        print(f"  Room {room:2d} | {tas_path.name} | {len(inputs)} inputs → {len(transitions)} transitions | skipped {skipped}")

    print("=" * 55)
    print(f"Total transitions: {len(all_transitions)}")
    print(f"Total skipped:     {total_skipped}")

    # Action distribution
    action_counts = {}
    for _, a in all_transitions:
        action_counts[a] = action_counts.get(a, 0) + 1

    print("\nAction distribution:")
    for idx, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        btn = SIMPLE_ACTIONS[idx]
        pct = 100 * count / len(all_transitions)
        print(f"  [{idx:2d}] btn={btn:2d}  {count:6d}  ({pct:.1f}%)")

    with open(output_path, 'wb') as f:
        pickle.dump(all_transitions, f)

    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
