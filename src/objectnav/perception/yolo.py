from __future__ import annotations

import warnings
import torch

from typing import List, Optional
import numpy as np

from ultralytics import YOLO

from .detections import Detection
from .patches import apply_ultralytics_patch


class YOLODetector:
    def __init__(self, weights_path: str, device: str = "cuda:0", conf: float = 0.25):
        self.weights_path = weights_path
        self.device = device
        self.conf = conf
        self._model: Optional[YOLO] = None
        self._resolved_device: Optional[str] = None

    def load(self) -> None:
        apply_ultralytics_patch()
        self._model = YOLO(self.weights_path)
        self._resolved_device = self._resolve_device()


    def detect_bgr(self, image_bgr: np.ndarray) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("YOLODetector not loaded. Call .load() first.")
        
        if self._resolved_device is None:
            self._resolved_device = self._resolve_device()

        # Ultralytics expects images as numpy arrays (BGR is typically okay; it will handle)
        results = self._model.predict(
            source=image_bgr,
            verbose=False,
            device=self._resolved_device,
            conf=self.conf,
        )

        # results is a list-like; take first image
        r0 = results[0]
        boxes = r0.boxes  # patched Boxes class should be active

        dets: List[Detection] = []
        if boxes is None or len(boxes) == 0:
            return dets

        # xyxy as numpy
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1)
        cls = boxes.cls.cpu().numpy().astype(int).reshape(-1)

        for i in range(len(cls)):
            dets.append(
                Detection(
                    cls=int(cls[i]),
                    conf=float(conf[i]),
                    xyxy=(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])),
                )
            )
        return dets
    
    def _resolve_device(self) -> str:
        # If user asked for CPU, respect it.
        if self.device.startswith("cpu"):
            return "cpu"

        # If CUDA isn't available at all, fall back.
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available; falling back to CPU for YOLO.")
            return "cpu"

        # CUDA exists, but the installed torch build may not support this GPU arch.
        # The most reliable check is to attempt a tiny CUDA op.
        try:
            _ = torch.zeros(1, device="cuda")
            return self.device
        except Exception as e:
            warnings.warn(f"CUDA appears unusable with this PyTorch build; falling back to CPU. ({e})")
            return "cpu"

