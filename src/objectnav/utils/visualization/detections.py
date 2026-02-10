from __future__ import annotations

from typing import Optional

def save_yolo_detections_plot(
    results: "Results",
    save_path: str,
    *,
    show_conf: bool = True,
    show_labels: bool = True,
    show_boxes: bool = True,
) -> None:
    """Save a plotted YOLO detections image to disk."""
    results.plot(
        conf=show_conf,
        labels=show_labels,
        boxes=show_boxes,
        save=True,
        filename=save_path,
    )
