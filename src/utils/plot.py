"""Shared training-curve plotting. One graph per run, labeled by run_id."""

from pathlib import Path
import numpy as np


def plot_run(run_id, rewards, heights, completions=None, stage_boundaries=None,
             save_dir='docs', window=50):
    """
    Save a labeled training curve to {save_dir}/{run_id}_curve.png.

    rewards, heights: per-episode lists
    completions: optional per-episode bool list (rolling completion rate panel)
    stage_boundaries: optional list of episode indices where curriculum stages change.
                      Vertical dashed lines drawn at each. Pairs of (idx, label) are
                      also accepted for stage labels.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    n_panels = 3 if completions is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
    if n_panels == 2:
        axes = list(axes)

    fig.suptitle(f'Training run: {run_id}', fontsize=14, fontweight='bold')

    # Reward
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(rewards)), ma, label=f'{window}-ep MA', linewidth=2)
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Height (96 - h: 0 = start, 96+ = exit / level complete)
    ax2 = axes[1]
    progress = [96 - h for h in heights]
    ax2.plot(progress, alpha=0.3, label='Progress (96 - height)')
    if len(progress) >= window:
        ma = np.convolve(progress, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(progress)), ma, label=f'{window}-ep MA', linewidth=2)
    ax2.axhline(y=100, color='g', linestyle=':', alpha=0.5, label='Level complete (~y=-4)')
    ax2.set_xlabel('Episode' if n_panels == 2 else '')
    ax2.set_ylabel('Progress up the level')
    ax2.set_title('Height Reached')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Completion rate (only if provided)
    if completions is not None:
        ax3 = axes[2]
        comp_int = [1 if c else 0 for c in completions]
        if len(comp_int) >= window:
            rate = np.convolve(comp_int, np.ones(window) / window, mode='valid')
            ax3.plot(range(window - 1, len(comp_int)), rate * 100, linewidth=2, color='tab:green')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Completion rate (%)')
        ax3.set_title(f'Rolling {window}-ep Completion Rate')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

    # Stage boundary markers (curriculum)
    if stage_boundaries:
        for ax in axes:
            for entry in stage_boundaries:
                if isinstance(entry, tuple):
                    idx, label = entry
                else:
                    idx, label = entry, None
                ax.axvline(x=idx, color='k', linestyle='--', alpha=0.4, linewidth=0.8)
                if label and ax is axes[0]:
                    ax.text(idx, ax.get_ylim()[1] * 0.95, label,
                            rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)

    plt.tight_layout()
    save_path = Path(save_dir) / f'{run_id}_curve.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curve to {save_path}")
    return str(save_path)
