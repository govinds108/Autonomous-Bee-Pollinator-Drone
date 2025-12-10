from djitellopy import Tello
import cv2
import numpy as np
from ultralytics import YOLO


def initializeTello():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    # Start the stream ONCE here
    drone.streamoff()
    drone.streamon()
    return drone


def telloGetFrame(drone, w=360, h=240):
    frame = drone.get_frame_read().frame
    return cv2.resize(frame, (w, h))


def analyzeFlower(img, x1, y1, x2, y2):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    area = (x2 - x1) * (y2 - y1)
    if area < 500:
        return 0.0

    s = hsv[:, :, 1]
    saturation_score = np.mean(s) / 130.0
    saturation_score = np.clip(saturation_score, 0, 1)

    h = hsv[:, :, 0]
    hue_variance = np.var(h)
    color_diversity = np.clip(hue_variance / 5000.0, 0, 1)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_score = np.clip(lap_var / 500.0, 0, 1)

    flower_score = (
        0.4 * saturation_score +
        0.3 * color_diversity +
        0.3 * texture_score
    )

    return float(flower_score)


def detectFlower(img, model, conf_thresh=0.25, flower_thresh=0.3):
    results = model(img, conf=conf_thresh, verbose=False)
    candidates = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            class_name = model.names[cls]
            if class_name != "potted plant":
                continue

            flower_score = analyzeFlower(img, x1, y1, x2, y2)

            if flower_score >= flower_thresh:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                candidates.append((x1, y1, x2, y2, cx, cy, area, conf, flower_score))

    if len(candidates) == 0:
        return img, [[0, 0], 0, 0, 0]

    candidates.sort(key=lambda x: x[8], reverse=True)
    x1, y1, x2, y2, cx, cy, area, conf, flower_score = candidates[0]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"Flower {flower_score:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    return img, [[cx, cy], area, conf, flower_score]


# ============================================================
# YAW-ONLY STATE (best for stable control)
# ============================================================
def get_state(cx, cy, area, w, h):
    # x error normalized to [-1, 1]
    x_err = (cx - w/2) / (w/2)
    # normalize area by image area to get a value in [0, 1]
    img_area = float(w * h)
    area_norm = float(area) / img_area if img_area > 0 else 0.0
    area_norm = np.clip(area_norm, 0.0, 1.0)
    # State now contains both centering error and normalized bounding-box area
    return np.array([x_err, area_norm], dtype=np.float32)
