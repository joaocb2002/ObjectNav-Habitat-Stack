from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def _normalize_rgb(rgb_obs: np.ndarray) -> np.ndarray:
    """Normalize RGB(A) observation to uint8 for visualization."""
    if rgb_obs.ndim != 3 or rgb_obs.shape[2] not in (3, 4):
        raise ValueError("rgb_obs must be an HxWx3 or HxWx4 array.")
    if np.issubdtype(rgb_obs.dtype, np.floating):
        # Accept [0, 1] or [0, 255] float ranges.
        max_val = float(np.nanmax(rgb_obs)) if rgb_obs.size else 0.0
        if max_val <= 1.0:
            rgb = np.clip(rgb_obs * 255.0, 0.0, 255.0)
        else:
            rgb = np.clip(rgb_obs, 0.0, 255.0)
        return rgb.astype(np.uint8)
    if np.issubdtype(rgb_obs.dtype, np.integer):
        return np.clip(rgb_obs, 0, 255).astype(np.uint8)
    raise ValueError("rgb_obs must be a float or integer array.")


def _normalize_depth(
    depth_obs: np.ndarray,
    *,
    depth_clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Normalize depth (meters) to uint8 image for visualization."""
    if depth_obs.ndim != 2:
        raise ValueError("depth_obs must be a 2D array.")
    if not np.issubdtype(depth_obs.dtype, np.floating):
        depth = depth_obs.astype(np.float32)
    else:
        depth = depth_obs

    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return np.zeros_like(depth, dtype=np.uint8)

    if depth_clip is not None:
        depth_min, depth_max = depth_clip
    else:
        depth_min = float(np.nanmin(depth))
        depth_max = float(np.nanmax(depth))

    if depth_max <= depth_min:
        return np.zeros_like(depth, dtype=np.uint8)

    depth = np.clip(depth, depth_min, depth_max)
    depth = (depth - depth_min) / (depth_max - depth_min)
    depth = np.clip(depth, 0.0, 1.0)
    return (depth * 255.0).astype(np.uint8)


def save_rgbd_observations(
    rgb_obs: np.ndarray,
    depth_obs: np.ndarray,
    *,
    save_path: str,
    figsize: Tuple[int, int] = (8, 5),
    show_axis: bool = False,
    depth_clip: Optional[Tuple[float, float]] = (0.0, 10.0),
    depth_cmap: str = "gray",
) -> None:
    """Save RGB and depth observations side-by-side to a file.

    Args:
        rgb_obs: RGB(A) image from the agent's RGB sensor (HxWx3 or HxWx4).
        depth_obs: Depth image as a 2D array of float distances in meters.
        save_path: Path where the figure will be saved.
        figsize: Figure size used when creating a new figure.
        show_axis: Whether to show axis ticks/labels. Default is False.
        depth_clip: Min/max depth range (meters) for visualization. If None, uses
            min/max in the observation.
        depth_cmap: Matplotlib colormap for depth display.
    """
    _, axes = plt.subplots(1, 2, figsize=figsize)

    rgb_vis = _normalize_rgb(rgb_obs)
    rgb_mode = "RGBA" if rgb_vis.shape[2] == 4 else "RGB"
    rgb_img = Image.fromarray(rgb_vis, mode=rgb_mode)
    axes[0].imshow(rgb_img)
    axes[0].set_title("rgb")
    if not show_axis:
        axes[0].axis("off")

    depth_vis = _normalize_depth(depth_obs, depth_clip=depth_clip)
    depth_img = Image.fromarray(depth_vis, mode="L")
    axes[1].imshow(depth_img, cmap=depth_cmap)
    axes[1].set_title("depth")
    if not show_axis:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()