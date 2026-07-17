"""
Shared state for the Whisper Fine-Tune GUI.

Provides a thread-safe ``TrainingState`` dataclass used across all tabs
to track training progress, logs, and stop events.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingState:
    """Thread-safe container for live training progress."""

    running: bool = False
    current_step: int = 0
    total_steps: int = 0
    current_loss: Optional[float] = None
    current_wer: Optional[float] = None
    current_lr: Optional[float] = None
    start_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    logs: List[str] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ── Thread-safe accessors ──────────────────────────────────────

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def reset_stop(self) -> None:
        self._stop_event.clear()

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "current_loss": self.current_loss,
                "current_wer": self.current_wer,
                "current_lr": self.current_lr,
                "elapsed_seconds": (
                    time.time() - self.start_time if self.start_time else 0.0
                ),
                "logs": self.logs[-50:],  # keep last 50 lines
            }

    def add_log(self, line: str) -> None:
        with self._lock:
            self.logs.append(line)
            # Keep at most 5000 lines to avoid memory bloat
            if len(self.logs) > 5000:
                self.logs = self.logs[-5000:]


# Global singleton — imported by all gui modules
training_state = TrainingState()
