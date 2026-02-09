from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from objectnav.perception.config import YoloConfig
from objectnav.perception.detections import Detection


class YOLODetector:
    def __init__(self, config: YoloConfig):
        self.config = config
        self._model: Optional[YOLO] = None
        self._resolved_device: Optional[str] = None

    def load(self) -> None:
        self._model = YOLO(self.config.weights_path_str())
        self._resolved_device = self._resolve_device()

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], "Results"]:
        """Run inference and parse detections from a single image."""
        if self._model is None:
            raise RuntimeError("YOLODetector not loaded. Call .load() first.")
        
        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()

        # Ultralytics expects images as numpy arrays (RGB/BGR is acceptable; it will handle)
        results = self._model.predict(
            source=image,
            verbose=self.config.verbose,
            device=self._resolved_device,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            rect=self.config.rect,
            half=self.config.half,
            max_det=self.config.max_det,
        )

        # results is a list-like; take first image
        r0 = results[0]
        detections = self._parse_results(r0, image)
        return detections, r0
    
    def _resolve_device(self) -> str:
        # If user asked for CPU, respect it.
        if self.config.device.startswith("cpu"):
            return "cpu"

        # If CUDA isn't available at all, fall back.
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available; falling back to CPU for YOLO.")
            return "cpu"

        # CUDA exists, but the installed torch build may not support this GPU arch.
        # The most reliable check is to attempt a tiny CUDA op.
        try:
            _ = torch.zeros(1, device="cuda")
            return self.config.device
        except Exception as e:
            warnings.warn(f"CUDA appears unusable with this PyTorch build; falling back to CPU. ({e})")
            return "cpu"

    def _parse_results(self, results: "Results", image: np.ndarray) -> List[Detection]:
        boxes = results.boxes  # patched Boxes class should be active
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1)
        cls = boxes.cls.cpu().numpy().astype(int).reshape(-1)
        # probs = boxes.probs
        # prob_vectors = probs.cpu().numpy() if probs is not None else None

        names = results.names if results.names is not None else {}
        image_area = float(image.shape[0] * image.shape[1])

        dets: List[Detection] = []
        for i in range(len(cls)):
            x1, y1, x2, y2 = (
                float(xyxy[i, 0]),
                float(xyxy[i, 1]),
                float(xyxy[i, 2]),
                float(xyxy[i, 3]),
            )
            det_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            scale = det_area / image_area if image_area > 0.0 else 0.0
            det_probs: Optional[Tuple[float, ...]]
            # if prob_vectors is not None:
            #     det_probs = tuple(float(p) for p in prob_vectors[i].tolist())
            # else:
            #     det_probs = None
            cls_id = int(cls[i])
            cls_name = str(names.get(cls_id, cls_id))
            dets.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=float(conf[i]),
                    xyxy=(x1, y1, x2, y2),
                    scale=scale,
                    # probs=det_probs,
                )
            )
        return dets
