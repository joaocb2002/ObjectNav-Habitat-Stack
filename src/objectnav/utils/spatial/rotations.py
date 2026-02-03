from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple, Union

import numpy as np

from objectnav.constants import CAMERA_DEFAULT_DIRECTION
from objectnav.types import QuaternionLike, ScalarLike, Vector3DLike

try:
    import quaternion  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency `numpy-quaternion` (module name: `quaternion`). "
        "Install it (e.g. `pip install numpy-quaternion`) or adjust this module "
        "to use a different quaternion type."
    ) from e



def _as_wxyz(q: QuaternionLike) -> Tuple[float, float, float, float]:
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


def coerce_quaternion(q: QuaternionLike, *, normalize: bool = True) -> "quaternion.quaternion":
    """
    Coerce a quaternion-like input to a quaternion.quaternion.

    Args:
        q: Quaternion input (quaternion.quaternion, wxyz sequence, or mapping).
        normalize: If True, normalize the quaternion.

    Returns:
        quaternion.quaternion instance.
    """
    if isinstance(q, quaternion.quaternion):
        if not normalize:
            return q
        w, x, y, z = _as_wxyz(q)
        w, x, y, z = _normalize_wxyz(w, x, y, z)
        return quaternion.quaternion(w, x, y, z)

    w, x, y, z = _as_wxyz(q)
    if normalize:
        w, x, y, z = _normalize_wxyz(w, x, y, z)
    return quaternion.quaternion(w, x, y, z)


def _normalize_wxyz(w: float, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    """Return a unit quaternion (w, x, y, z)."""
    n2 = w * w + x * x + y * y + z * z
    if n2 == 0.0:
        raise ValueError("Cannot normalize a zero-norm quaternion.")
    inv_n = 1.0 / math.sqrt(n2)
    return w * inv_n, x * inv_n, y * inv_n, z * inv_n


def quaternion_to_yaw(q: QuaternionLike, *, degrees: bool = True, normalize: bool = True) -> float:
    """
    Convert a quaternion to a yaw/heading angle.

    Convention:
      - Assumes a Y-up world where "yaw" is rotation about the +Y axis 
      in a right-handed system.
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


def yaw_to_quaternion(yaw: ScalarLike, *, degrees: bool = True) -> "quaternion.quaternion":
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


def rotate_vector_by_quaternion(
    v: Vector3DLike,
    q: QuaternionLike,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Rotate a 3D vector by a quaternion.

    Args:
        v: 3D vector (x, y, z).
        q: Quaternion input in (w, x, y, z) format.
        normalize: If True, normalizes the quaternion before rotation and
            normalizes the resulting vector.

    Returns:
        Rotated vector as a numpy array of shape (3,).
    """
    v_arr = np.asarray(v, dtype=np.float64)
    if v_arr.shape != (3,):
        raise ValueError(f"v must be length 3, got shape {v_arr.shape}")

    w, x, y, z = _as_wxyz(q)
    if normalize:
        w, x, y, z = _normalize_wxyz(w, x, y, z)

    # Quaternion-vector rotation: v' = q * v * q_conj
    # Use the optimized formula to avoid constructing full quaternions.
    q_vec = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(q_vec, v_arr)
    v_rot = v_arr + w * t + np.cross(q_vec, t)

    if not normalize:
        return v_rot
    n = float(np.linalg.norm(v_rot))
    if n == 0.0:
        raise ValueError("Rotated vector has zero norm; cannot normalize.")
    return v_rot / n


def facing_direction_from_rotation(
    rotation: QuaternionLike,
    *,
    initial_forward: Vector3DLike = CAMERA_DEFAULT_DIRECTION,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the world-space facing direction from a quaternion rotation.

    Args:
        rotation: Quaternion in (w, x, y, z) format.
        initial_forward: Local forward direction to rotate (default: camera forward).
        normalize: Normalize the resulting direction vector.

    Returns:
        World-space facing direction as a numpy array of shape (3,).
    """
    return rotate_vector_by_quaternion(initial_forward, rotation, normalize=normalize)


def direction_to_yaw(direction: Vector3DLike, *, degrees: bool = True) -> float:
    """
    Convert a world-space facing direction vector to a yaw/heading angle.

        Convention:
            - Y-up world; yaw is rotation about +Y axis.
            - Yaw = 0 when facing +Z, increases toward +X (right-handed).

    Args:
        direction: World-space facing direction (x, y, z).
        degrees: If True returns degrees, else radians.

    Returns:
        Yaw angle as float (degrees by default; radians if degrees=False).
    """
    d = np.asarray(direction, dtype=np.float64)
    if d.shape != (3,):
        raise ValueError(f"direction must be length 3, got shape {d.shape}")
    if np.allclose(d, 0.0):
        raise ValueError("direction must be non-zero to compute yaw")

    # For Y-up, yaw = atan2(x, z) -> yaw=0 when facing +Z.
    yaw = math.atan2(float(d[0]), float(d[2]))
    return math.degrees(yaw) if degrees else yaw


def normalize_yaw(yaw: ScalarLike, *, degrees: bool = True) -> float:
    """
    Normalize a yaw angle to a canonical range.

    Args:
        yaw: Input yaw angle.
        degrees: If True, normalize to [-180, 180); else to [-pi, pi).

    Returns:
        Normalized yaw angle.
    """
    if degrees:
        return ((float(yaw) + 180.0) % 360.0) - 180.0
    return ((float(yaw) + math.pi) % (2.0 * math.pi)) - math.pi


def rotation_to_yaw(
    rotation: Union[QuaternionLike, ScalarLike],
    *,
    degrees: bool = True,
    scalar_is_degrees: bool = True,
    scalar_offset_degrees: float = 0.0,
    initial_forward: Vector3DLike = CAMERA_DEFAULT_DIRECTION,
    normalize: bool = True,
) -> float:
    """
    Convert a rotation (scalar yaw or quaternion) to a yaw/heading angle.
    Serves as a unified interface for both input types.

    Args:
        rotation: Either a scalar yaw (float/int) or a quaternion-like input.
        degrees: If True returns degrees, else radians.
        scalar_is_degrees: If rotation is scalar, interpret it as degrees when True,
            otherwise radians.
        scalar_offset_degrees: Offset to apply to scalar yaw (in degrees).
        initial_forward: Local forward direction to rotate when rotation is a quaternion.

    Returns:
        Yaw angle as float (degrees by default; radians if degrees=False).
    """
    if isinstance(rotation, (int, float, np.floating)):
        yaw_deg = float(rotation)
        if not scalar_is_degrees:
            yaw_deg = math.degrees(yaw_deg)
        yaw_deg = yaw_deg + float(scalar_offset_degrees)
        yaw = yaw_deg if degrees else math.radians(yaw_deg)
        return normalize_yaw(yaw, degrees=degrees) if normalize else yaw

    facing = facing_direction_from_rotation(rotation, initial_forward=initial_forward, normalize=True)
    yaw = direction_to_yaw(facing, degrees=degrees)
    return normalize_yaw(yaw, degrees=degrees) if normalize else yaw
