from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from .state import BeliefState


@dataclass
class BeliefUpdater:
    """
    Wrapper around legacy belief update logic.

    For now this delegates to legacy code. Later we move logic here.
    """
    # Put tunable parameters here later (thresholds, priors, etc.)
    config: Optional[dict] = None

    def init_state(self, shape: tuple[int, int, int]) -> BeliefState:
        # Example: (H, W, C) categorical distribution per cell
        grid = np.zeros(shape, dtype=np.float32)
        return BeliefState(grid=grid, meta={"init_shape": shape})

    def update(self, state: BeliefState, *args: Any, **kwargs: Any) -> BeliefState:
        # Implement in step 5.3 by calling legacy update function(s)
        raise NotImplementedError
