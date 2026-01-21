from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class BeliefState:
    """
    Minimal belief state.

    We start small and expand later.
    """
    # Example: a grid map of categorical probs or counts
    grid: np.ndarray

    # Optional extras (add later as you migrate)
    meta: Optional[dict] = None
