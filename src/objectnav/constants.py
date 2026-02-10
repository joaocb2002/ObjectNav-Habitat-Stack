"""Global immutable constants for object navigation.

These values are intended to be stable across experiments. Runtime-configurable
values should live in YAML configs under configs/ or in dataclass configs.
"""

from __future__ import annotations

from typing import Final, Tuple, FrozenSet

# --- Coordinate/frame constants ---
# Local agent forward direction in Habitat (default): -Z.
CAMERA_DEFAULT_DIRECTION: Final[Tuple[float, float, float]] = (0.0, 0.0, -1.0)
AGENT_DEFAULT_DIRECTION: Final[Tuple[float, float, float]] = (0.0, 0.0, 1.0)
CAMERA_DEFAULT_YAW_OFFSET_DEGREES: Final[float] = 180.0  # to convert from camera forward to agent forward


# --- Simulation parameters ---
CONFIDENCE_THRESHOLD: Final[float] = 0.80
LOCATION_ERROR_THRESHOLD: Final[float] = 0.5  # meters
MAX_ITER_COEF: Final[float] = 0.75  # coefficient to determine max iterations
PSEUDO_COUNT_THRESHOLD: Final[float] = 6.0
NUM_EPOCHS: Final[int] = 100

# --- Action sets ---
EXTENDED_ACTIONS: Final[FrozenSet[str]] = frozenset(
    {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_around",
        "turn_left",
        "turn_right",
    }
)

ACTIONS: Final[FrozenSet[str]] = frozenset(
    {
        "move_forward",
        "turn_left",
        "turn_right",
    }
)

# --- Probabilistic model parameters ---
DIRICHLET_PRIOR: Final[float] = 1.0

# --- Display settings ---
DISPLAY_STEP: Final[int] = 10  # display every N actions by default

__all__ = [
    "ACTIONS",
    "CONFIDENCE_THRESHOLD",
    "DIRICHLET_PRIOR",
    "DISPLAY_STEP",
    "EXTENDED_ACTIONS",
    "CAMERA_DEFAULT_DIRECTION",
    "AGENT_DEFAULT_DIRECTION",
    "LOCATION_ERROR_THRESHOLD",
    "MAX_ITER_COEF",
    "NUM_EPOCHS",
    "PSEUDO_COUNT_THRESHOLD",
    "CAMERA_DEFAULT_YAW_OFFSET_DEGREES",
]