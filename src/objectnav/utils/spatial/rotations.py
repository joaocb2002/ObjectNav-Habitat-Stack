from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple, Union, overload

try:
    import quaternion  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency `numpy-quaternion` (module name: `quaternion`). "
        "Install it (e.g. `pip install numpy-quaternion`) or adjust this module "
        "to use a different quaternion type."
    ) from e

WXYZLike = Union["quaternion.quaternion", Sequence[float], Mapping[str, float]]


def _as_wxyz(q: WXYZLike) -> Tuple[float, float, float, float]:
    """
    Coerce a quaternion-like input to (w, x, y, z).

    Accepts:
      - quaternion.quaternion (attributes: w, x, y, z)
      - Sequence[float] of length 4: (w, x, y, z)
      - Mapping with keys {'w','x','y','z'}
    """
    if hasattr(q, "w") and hasattr(q, "x") and hasattr(q, "y") and hasattr(q, "z"):
        return float(q.w), float(q.x), float(q.y), float(q.z)

    if isinstance(q, Mapping):
        return float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])

    if isinstance(q, Sequence) and len(q) == 4:
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    raise TypeError(
        "q must be a quaternion.quaternion, a length-4 sequence (w,x,y,z), "
        "or a mapping with keys w,x,y,z."
    )


def _normalize_wxyz(w: float, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    """Return a unit quaternion (w, x, y, z)."""
    n2 = w * w + x * x + y * y + z * z
    if n2 == 0.0:
        raise ValueError("Cannot normalize a zero-norm quaternion.")
    inv_n = 1.0 / math.sqrt(n2)
    return w * inv_n, x * inv_n, y * inv_n, z * inv_n


def quaternion_to_yaw(q: WXYZLike, *, degrees: bool = True, normalize: bool = True) -> float:
    """
    Convert a quaternion to a yaw/heading angle.

    Convention:
      - Assumes a Y-up world (common in graphics/Habitat) where "yaw" is rotation
        about the +Y axis.
      - Quaternion is assumed in (w, x, y, z) scalar-first format.

    Args:
        q: Quaternion input. Supported:
           - quaternion.quaternion with attributes w,x,y,z
           - length-4 sequence (w,x,y,z)
           - mapping with keys "w","x","y","z"
        degrees: If True returns degrees, else radians.
        normalize: If True, normalizes the quaternion before extracting yaw.

    Returns:
        Yaw angle as float (degrees by default; radians if degrees=False).
    """
    w, x, y, z = _as_wxyz(q)
    if normalize:
        w, x, y, z = _normalize_wxyz(w, x, y, z)

    # Heading about Y axis for Y-up coordinates.
    # For a pure yaw quaternion (cos(a), 0, sin(a), 0), this yields yaw = 2a.
    siny_cosp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(yaw) if degrees else yaw


def yaw_to_quaternion(yaw: float, *, degrees: bool = True) -> "quaternion.quaternion":
    """
    Convert a yaw/heading angle to a quaternion.

    Convention:
      - Y-up world; yaw is a rotation about +Y axis.
      - Output is scalar-first quaternion (w, x, y, z).

    Args:
        yaw: Yaw angle (degrees by default; radians if degrees=False).
        degrees: Interpret `yaw` as degrees if True, else radians.

    Returns:
        quaternion.quaternion representing a pure yaw rotation about +Y.
    """
    yaw_rad = math.radians(yaw) if degrees else float(yaw)
    half = 0.5 * yaw_rad
    return quaternion.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)