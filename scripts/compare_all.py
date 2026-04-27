"""
Run a controlled evaluation across every model in runs/, then save comparison plots.

Default behavior: discover every `runs/*/best.pt` and evaluate it with ε=0.05.
Use --override RUN_ID=PATH to point at a different checkpoint for a specific run
(useful when best.pt was saved early and the real peak is later — see final_log.md).

Usage:
    python scripts/compare_all.py
    python scripts/compare_all.py --episodes 100 --epsilon 0.05
    python scripts/compare_all.py \
        --override dqn_r1=runs/dqn_r1/checkpoint_ep5000.pt \
        --override v3_r9=runs/v3_r9/checkpoint_ep2000.pt \
        --override v3_r8=runs/v3_r8/checkpoint_ep3000.pt \
        --override hybrid_r2=runs/hybrid_r2/checkpoint_ep5000.pt

Outputs (under docs/comparison/):
    comparison_table.csv         per-run summary
    completion_bar.png           bar chart of completion rate
    outcome_breakdown.png        stacked bar of complete/died/timeout
    height_distribution.png      mean height per run
    summary.json                 raw numbers for further analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import CelesteEnv
from src.agent import DQNAgent
from src.network import DuelingDQN

from scripts.evaluate import detect_architecture


def evaluate_model(model_path, room=0, num_episodes=100, epsilon=0.05, max_steps=500):
    """Run a single model evaluation, return outcome dict."""
    env = CelesteEnv(room=room, max_steps=max_steps)

    arch = detect_architecture(model_path)
    dueling = (arch == 'dueling')

    agent = DQNAgent(
        state_dim=env._get_obs_dim(),
        action_dim=env.n_actions,
        device='cpu',
        network_cls=DuelingDQN if dueling else None,
    )
    agent.load(model_path)
    agent.epsilon = epsilon

    rewards, heights = [], []
    completes = deaths = timeouts = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        while True:
            action = agent.select_action(state, training=(epsilon > 0))
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        heights.append(info['max_height'])
        if info.get('completed', False):
            completes += 1
        elif not info['player_alive']:
            deaths += 1
        else:
            timeouts += 1

    return {
        'arch': arch,
        'episodes': num_episodes,
        'epsilon': epsilon,
        'completes': completes,
        'deaths': deaths,
        'timeouts': timeouts,
        'completion_rate': completes / num_episodes,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_height': float(np.mean(heights)),
        'min_height': float(np.min(heights)),
        'max_height': float(np.max(heights)),
    }


def evaluate_random(room=0, num_episodes=100, max_steps=500):
    """Random-action baseline."""
    env = CelesteEnv(room=room, max_steps=max_steps)
    rewards, heights = [], []
    completes = deaths = timeouts = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        while True:
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        heights.append(info['max_height'])
        if info.get('completed', False):
            completes += 1
        elif not info['player_alive']:
            deaths += 1
        else:
            timeouts += 1
    return {
        'arch': 'random',
        'episodes': num_episodes,
        'epsilon': None,
        'completes': completes,
        'deaths': deaths,
        'timeouts': timeouts,
        'completion_rate': completes / num_episodes,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_height': float(np.mean(heights)),
        'min_height': float(np.min(heights)),
        'max_height': float(np.max(heights)),
    }


def discover_runs(runs_dir='runs', overrides=None):
    """Find models in runs/, supporting two layouts:
    - runs/{run_id}/best.pt          (per-run folder)
    - runs/{run_id}.pt               (flat, matches release naming)
    Overrides take precedence.
    """
    overrides = overrides or {}
    runs = {}
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return runs

    # Folder layout: runs/{run_id}/best.pt
    for sub in sorted(runs_path.iterdir()):
        if sub.is_dir():
            best = sub / 'best.pt'
            if best.exists():
                runs[sub.name] = str(best)

    # Flat layout: runs/{run_id}.pt
    for f in sorted(runs_path.glob('*.pt')):
        run_id = f.stem
        runs.setdefault(run_id, str(f))

    # Apply overrides
    for run_id, path in overrides.items():
        runs[run_id] = path

    return runs


def plot_comparison(results, save_dir):
    """Save bar/breakdown/height plots into save_dir."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # Sort by completion rate descending so the winner is leftmost
    items = sorted(results.items(), key=lambda kv: -kv[1]['completion_rate'])
    names = [k for k, _ in items]
    completes = [v['completion_rate'] * 100 for _, v in items]
    deaths = [v['deaths'] / v['episodes'] * 100 for _, v in items]
    timeouts = [v['timeouts'] / v['episodes'] * 100 for _, v in items]
    mean_h = [v['mean_height'] for _, v in items]

    # 1. Completion rate bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, completes, color='tab:green')
    ax.set_ylabel('Completion rate (%)')
    ax.set_title('Celeste room 0 — completion rate by method (ε=0.05, 100 ep)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, completes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'completion_bar.png', dpi=150)
    plt.close(fig)

    # 2. Outcome breakdown — stacked bar (complete / died / timeout)
    fig, ax = plt.subplots(figsize=(10, 5))
    ind = np.arange(len(names))
    ax.bar(ind, completes, color='tab:green', label='Complete')
    ax.bar(ind, deaths, bottom=completes, color='tab:red', label='Died')
    bottoms = [c + d for c, d in zip(completes, deaths)]
    ax.bar(ind, timeouts, bottom=bottoms, color='tab:gray', label='Timeout')
    ax.set_xticks(ind)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Episodes (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Outcome breakdown by method (ε=0.05, 100 ep)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'outcome_breakdown.png', dpi=150)
    plt.close(fig)

    # 3. Mean height per run (lower = better, since y=0 is exit)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, mean_h, color='tab:blue')
    ax.axhline(y=-4, color='g', linestyle='--', alpha=0.5, label='Exit threshold (y < -4)')
    ax.set_ylabel('Mean min-y reached')
    ax.set_title('Mean height per method (lower = closer to exit)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir / 'height_distribution.png', dpi=150)
    plt.close(fig)

    print(f"Plots saved to {save_dir}/")


def write_csv(results, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['run_id', 'arch', 'episodes', 'completes', 'deaths', 'timeouts',
            'completion_rate', 'mean_reward', 'std_reward', 'mean_height']
    lines = [','.join(cols)]
    for run_id, v in sorted(results.items(), key=lambda kv: -kv[1]['completion_rate']):
        row = [run_id, v['arch'], v['episodes'], v['completes'], v['deaths'],
               v['timeouts'], f"{v['completion_rate']:.3f}",
               f"{v['mean_reward']:.2f}", f"{v['std_reward']:.2f}",
               f"{v['mean_height']:.2f}"]
        lines.append(','.join(str(x) for x in row))
    save_path.write_text('\n'.join(lines))


def parse_overrides(override_args):
    """--override dqn_r1=path/to/file.pt -> dict."""
    overrides = {}
    for arg in override_args or []:
        if '=' not in arg:
            raise ValueError(f"--override expects RUN_ID=PATH, got: {arg}")
        run_id, path = arg.split('=', 1)
        overrides[run_id.strip()] = path.strip()
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', type=str, default='runs')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--include-random', action='store_true', default=True,
                        help='Include random-action baseline (default: on)')
    parser.add_argument('--no-random', dest='include_random', action='store_false')
    parser.add_argument('--override', action='append', default=[],
                        help='Per-run checkpoint override: --override dqn_r1=runs/dqn_r1/checkpoint_ep5000.pt')
    parser.add_argument('--output-dir', type=str, default='docs/comparison')
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    runs = discover_runs(args.runs_dir, overrides)

    if not runs:
        print(f"No runs found under {args.runs_dir}/. Did you train any?")
        return

    print(f"Will evaluate {len(runs)} runs at ε={args.epsilon}, {args.episodes} episodes each:\n")
    for run_id, path in runs.items():
        print(f"  {run_id:20s} -> {path}")
    print()

    results = {}

    if args.include_random:
        print("Random baseline...")
        results['random'] = evaluate_random(args.room, args.episodes, args.max_steps)
        print(f"  -> {results['random']['completion_rate']*100:.1f}% completion\n")

    for run_id, path in runs.items():
        print(f"Evaluating {run_id} ({path})...")
        try:
            results[run_id] = evaluate_model(
                path, room=args.room, num_episodes=args.episodes,
                epsilon=args.epsilon, max_steps=args.max_steps,
            )
            r = results[run_id]
            print(f"  -> {r['completion_rate']*100:.1f}% completion  "
                  f"({r['completes']}/{r['episodes']})  "
                  f"arch={r['arch']}  mean_reward={r['mean_reward']:.1f}\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")

    # Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    write_csv(results, out / 'comparison_table.csv')
    plot_comparison(results, out)

    # Summary table
    print('=' * 70)
    print(f"{'Run':20s} {'Arch':10s} {'Complete':>10s} {'Died':>6s} {'Timeout':>8s}")
    print('-' * 70)
    for run_id, v in sorted(results.items(), key=lambda kv: -kv[1]['completion_rate']):
        print(f"{run_id:20s} {v['arch']:10s} "
              f"{v['completes']:>4}/{v['episodes']:<4} ({v['completion_rate']*100:>4.1f}%)  "
              f"{v['deaths']:>4}   {v['timeouts']:>5}")
    print('=' * 70)
    print(f"\nResults saved to {out}/")


if __name__ == '__main__':
    main()
