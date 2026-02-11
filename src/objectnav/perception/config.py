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
    conf: float = field(default=0.45, metadata={"help": "Confidence threshold."})
    iou: float = field(default=0.35, metadata={"help": "IoU threshold for NMS."})
    imgsz: Tuple[int, int] = field(
        default=(576, 1024),
        metadata={"help": "Image (height, width) for inference."},
    )
    rect: bool = field(default=True, metadata={"help": "Enable rectangular inference."})
    half: bool = field(default=False, metadata={"help": "Use half precision when supported."})
    max_det: int = field(default=30, metadata={"help": "Maximum detections per image."})
    verbose: bool = field(default=False, metadata={"help": "Enable per-inference logging."})

    # --- Patches ---
    use_softmax_patch: bool = field(
        default=False,
        metadata={"help": "Apply softmax patch to expose class probabilities."},
    )
    softmax_temperature: float = field(
        default=2.4,
        metadata={"help": "Softmax temperature for class probabilities."},
    )

    def weights_path_str(self) -> str:
        return str(self.weights_path)
