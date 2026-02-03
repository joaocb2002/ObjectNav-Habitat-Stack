"""Shared typing aliases used across the project."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import quaternion  # type: ignore
    import magnum as mn  # type: ignore

QuaternionLike = Union["quaternion.quaternion", Sequence[float], Mapping[str, float]]
ScalarLike = Union[int, float, np.floating]
Vector3Like = Union[Sequence[float], np.ndarray, "mn.Vector3", "magnum.Vector3"]
PositionLike = Vector3Like
Position3D = Union[Sequence[float], np.ndarray]
GridCoord = Tuple[int, int]

__all__ = [
    "GridCoord",
    "Position3D",
    "PositionLike",
    "QuaternionLike",
    "ScalarLike",
    "Vector3Like",
]