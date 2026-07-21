"""
Utility functions for the Whisper Fine-Tune GUI.

- ``list_checkpoints()`` — enumerate available model checkpoints
- ``read_tb_scalar()`` — poll TensorBoard event files for live charts
- ``format_time()`` — human-readable duration
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def list_checkpoints(checkpoints_dir: str = "models/checkpoints") -> List[str]:
    """
    Scan *checkpoints_dir* for subdirectories matching ``checkpoint-NNNN``
    or a final model (no numeric suffix).

    Returns a list of checkpoint labels sorted by step, with the best
    model listed first if ``load_best_model_at_end`` was used.
    """
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.exists():
        return ["No checkpoints found"]

    checkpoints: List[Tuple[int, str]] = []

    for subdir in ckpt_dir.iterdir():
        if not subdir.is_dir():
            continue
        name = subdir.name

        # Check for trainer state to confirm it's a valid checkpoint
        if not (subdir / "trainer_state.json").exists():
            # Might still be a valid model directory (final model without state)
            if not (subdir / "config.json").exists():
                continue

        # Determine step number
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                checkpoints.append((step, name))
            except (IndexError, ValueError):
                checkpoints.append((0, name))
        else:
            # Final model directory — sort to top
            checkpoints.append((0, name))

    # Sort: larger step first (most recent)
    checkpoints.sort(key=lambda x: -x[0])
    return [name for _, name in checkpoints] or ["No checkpoints found"]


def read_tb_scalar(
    log_dir: str,
    tag: str,
    max_points: int = 200,
) -> Tuple[List[float], List[float]]:
    """
    Read a scalar *tag* (e.g., ``"eval/wer"``, ``"train/loss"``) from
    TensorBoard event files under *log_dir*.

    Returns ``(steps, values)``, each trimmed to *max_points* for
    plotting performance.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return [], []

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        ea = EventAccumulator(str(log_path))
        ea.Reload()

        if tag not in ea.Tags().get("scalars", []):
            return [], []

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # Downsample if too many points
        if len(steps) > max_points:
            indices = np.linspace(0, len(steps) - 1, max_points, dtype=int)
            steps = [steps[i] for i in indices]
            values = [values[i] for i in indices]

        return steps, values

    except ImportError:
        return [], []
    except Exception:
        return [], []


def read_training_metrics(log_dir: str) -> Dict[str, dict]:
    """
    Read all available training metrics from TensorBoard logs.

    Returns a dict like::

        {
            "train/loss": {"steps": [...], "values": [...]},
            "eval/wer": {"steps": [...], "values": [...]},
            ...
        }
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        ea = EventAccumulator(str(log_path))
        ea.Reload()
        scalar_tags = ea.Tags().get("scalars", [])

        metrics = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            metrics[tag] = {
                "steps": [e.step for e in events],
                "values": [e.value for e in events],
            }
        return metrics

    except ImportError:
        return {}
    except Exception:
        return {}


def format_time(seconds: float) -> str:
    """Format *seconds* as ``"5m 23s"`` or ``"1h 12m 34s"``."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config, returning empty dict on failure."""
    import yaml
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def save_config(config_path: str, config: dict) -> None:
    """Save *config* dict to *config_path* as YAML."""
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
