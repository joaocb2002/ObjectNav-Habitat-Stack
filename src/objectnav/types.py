"""Shared typing aliases used across the project."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import quaternion  # type: ignore
    import magnum as mn  # type: ignore

QuaternionLike = Union["quaternion.quaternion", Sequence[float], Mapping[str, float]]
ScalarLike = Union[int, float, np.floating]
Vector3DLike = Union[Sequence[float], np.ndarray, "mn.Vector3"]
Position3DLike = Union[Sequence[float], np.ndarray]
Grid2DCoord = Tuple[int, int]

__all__ = [
    "QuaternionLike",
    "ScalarLike",
    "Vector3DLike",
    "Position3DLike",
    "Grid2DCoord",
]