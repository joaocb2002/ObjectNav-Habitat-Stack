from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import YoloConfig
from .detections import Detection
from .yolo import YOLODetector

def build_yolo_detector(config: YoloConfig) -> YOLODetector:
	"""Create and load a YOLO detector from configuration."""
	detector = YOLODetector(config)
	detector.load()
	return detector


def run_yolo_inference(
	detector: YOLODetector,
	image: np.ndarray,
	*,
	input_color: str = "rgb",
) -> Tuple[List[Detection], "Results"]:
	"""Run YOLO inference and return detections plus raw results."""
	image_bgr = _ensure_bgr(image, input_color=input_color)
	return detector.detect(image_bgr)


def _ensure_bgr(image: np.ndarray, *, input_color: str) -> np.ndarray:
	if input_color not in {"rgb", "bgr"}:
		raise ValueError("input_color must be 'rgb' or 'bgr'.")
	if image.ndim != 3 or image.shape[2] != 3:
		raise ValueError("image must be an HxWx3 array.")
	if input_color == "bgr":
		return image
	return np.ascontiguousarray(image[:, :, ::-1])
