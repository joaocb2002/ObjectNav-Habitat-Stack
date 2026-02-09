from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class YoloConfig:
    """Configuration for YOLO-based perception.

    Prefer overriding via a run config (YAML) rather than editing code.
    """

    # --- Model parameters ---
    weights_path: Union[str, Path] = field(
        default=Path("datasets/models/yolo11x.pt"),
        metadata={"help": "Path to YOLO weights file."},
    )
    device: str = field(default="cuda:0", metadata={"help": "Torch device specifier."})

    # --- Inference parameters ---
    conf: float = field(default=0.25, metadata={"help": "Confidence threshold."})
    iou: float = field(default=0.50, metadata={"help": "IoU threshold for NMS."})
    imgsz: Optional[Tuple[int, int]] = field(
        default=None,
        metadata={"help": "(height, width) for inference. None uses model default."},
    )
    rect: bool = field(default=True, metadata={"help": "Enable rectangular inference."})
    half: bool = field(default=False, metadata={"help": "Use half precision when supported."})
    max_det: int = field(default=30, metadata={"help": "Maximum detections per image."})
    verbose: bool = field(default=False, metadata={"help": "Enable per-inference logging."})

    def weights_path_str(self) -> str:
        return str(self.weights_path)
