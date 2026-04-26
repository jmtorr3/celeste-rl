"""Run-artifact path helpers. All outputs for a run live under runs/{run_id}/."""

from pathlib import Path


def run_dir(run_id: str, base: str = 'runs') -> Path:
    """Return runs/{run_id}/, creating it if missing."""
    p = Path(base) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p
