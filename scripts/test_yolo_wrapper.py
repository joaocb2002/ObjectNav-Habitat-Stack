import os
import numpy as np
from objectnav.perception.yolo import YOLODetector

def main():

    weights = os.environ.get("YOLO_WEIGHTS")
    if not weights:
        raise SystemExit("Set YOLO_WEIGHTS to your model path, e.g. /data/models/yolo11x.pt")

    det = YOLODetector(weights_path=weights, device="cuda:0", conf=0.25)
    det.load()

    print("YOLO device:", getattr(det, "_resolved_device", None))

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.detect_bgr(img)
    print("YOLO wrapper OK. Detections:", len(dets))

if __name__ == "__main__":
    main()
