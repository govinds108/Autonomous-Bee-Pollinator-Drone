import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    ok: bool
    name: Optional[str]
    conf: float
    xyxy: Optional[Tuple[float, float, float, float]]


def discretize_bbox_state(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    n_bins: int = 10,
) -> Tuple[int, int, int, float]:
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0

    if cx < img_w / 3.0:
        region = 0
    elif cx < 2.0 * img_w / 3.0:
        region = 1
    else:
        region = 2

    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    closeness = area / float(img_w * img_h)

    b = int(math.floor(closeness * n_bins))
    b = max(0, min(n_bins - 1, b))

    state_id = region * n_bins + b
    return state_id, region, b, float(closeness)


class YoloPerception:
    def __init__(
        self,
        model_path: str,
        target_class: Optional[str] = None,
        conf_thresh: float = 0.25,
        device: str = "cpu",
        imgsz: int = 320,
    ):
        self.model = YOLO(model_path)
        self.target_class = target_class
        self.conf_thresh = float(conf_thresh)
        self.device = device
        self.imgsz = int(imgsz)

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        r = self.model.predict(frame_bgr, conf=self.conf_thresh, verbose=False, device=self.device, imgsz=self.imgsz)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return Detection(ok=False, name=None, conf=0.0, xyxy=None)

        best_xyxy = None
        best_name = None
        best_conf = -1.0

        best_any_xyxy = None
        best_any_name = None
        best_any_conf = -1.0

        for b in r.boxes:
            cls_id = int(b.cls[0])
            name = self.model.names.get(cls_id, str(cls_id))
            conf = float(b.conf[0])

            if conf > best_any_conf:
                best_any_conf = conf
                best_any_name = name
                best_any_xyxy = tuple(float(v) for v in b.xyxy[0].tolist())

            if self.target_class is not None and name != self.target_class:
                continue

            if conf > best_conf:
                best_conf = conf
                best_name = name
                best_xyxy = tuple(float(v) for v in b.xyxy[0].tolist())

        if best_xyxy is None and best_any_xyxy is not None:
            best_xyxy = best_any_xyxy
            best_name = best_any_name
            best_conf = best_any_conf

        if best_xyxy is None:
            return Detection(ok=False, name=None, conf=0.0, xyxy=None)

        return Detection(ok=True, name=best_name, conf=float(best_conf), xyxy=best_xyxy)
