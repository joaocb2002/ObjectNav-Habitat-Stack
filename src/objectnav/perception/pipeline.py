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
) -> "Results":
	"""Run YOLO inference and return raw results."""
	image_rgb_or_bgr = _ensure_3_channel(image)
	image_bgr = _ensure_bgr(image_rgb_or_bgr, input_color=input_color)
	return detector.detect(image_bgr)


def run_yolo_detections(
	detector: YOLODetector,
	image: np.ndarray,
	*,
	input_color: str = "rgb",
) -> Tuple[List[Detection], "Results"]:
	"""Run YOLO and return parsed detections plus raw results."""
	yolo_results = run_yolo_inference(detector, image, input_color=input_color)
	detections = detector.parse_detections(yolo_results, image)
	return detections, yolo_results


def _ensure_3_channel(image: np.ndarray) -> np.ndarray:
	if image.ndim != 3:
		raise ValueError("image must be an HxWxC array.")
	if image.shape[2] == 3:
		return image
	if image.shape[2] == 4:
		return np.ascontiguousarray(image[:, :, :3])
	raise ValueError("image must have 3 channels (RGB/BGR) or 4 channels (RGBA/BGRA).")


def _ensure_bgr(image: np.ndarray, *, input_color: str) -> np.ndarray:
	if input_color not in {"rgb", "bgr"}:
		raise ValueError("input_color must be 'rgb' or 'bgr'.")
	if image.ndim != 3 or image.shape[2] != 3:
		raise ValueError("image must be an HxWx3 array.")
	if input_color == "bgr":
		return image
	return np.ascontiguousarray(image[:, :, ::-1])
