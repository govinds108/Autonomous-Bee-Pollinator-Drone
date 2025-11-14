from djitellopy import Tello
import cv2
import numpy as np
from ultralytics import YOLO


def initializeTello():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")

    drone.streamoff()
    drone.streamon()
    return drone


def telloGetFrame(drone, w=360, h=240):
    frame = drone.get_frame_read().frame
    return cv2.resize(frame, (w, h))


def analyzeFlower(img, x1, y1, x2, y2):
    """
    Flower likelihood score (0–1)
    """
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Basic filters
    area = (x2 - x1) * (y2 - y1)
    if area < 500:
        return 0.0

    # Saturation score
    s = hsv[:, :, 1]
    saturation_score = np.mean(s) / 130.0
    saturation_score = np.clip(saturation_score, 0, 1)

    # Color diversity
    h = hsv[:, :, 0]
    hue_variance = np.var(h)
    color_diversity = np.clip(hue_variance / 5000.0, 0, 1)

    # Texture
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_score = np.clip(lap_var / 500.0, 0, 1)

    # Combine
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
            print("Detected:", class_name, "conf:", conf)

            # STRICT FILTER — ONLY potted plant allowed
            if class_name != "potted plant":
                continue

            # Compute flower heuristic inside potted plant region
            flower_score = analyzeFlower(img, x1, y1, x2, y2)

            if flower_score >= flower_thresh:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                candidates.append((x1, y1, x2, y2, cx, cy, area, conf, flower_score))

    # No flower found
    if len(candidates) == 0:
        return img, [[0, 0], 0, 0, 0]

    # Choose highest flower_score
    candidates.sort(key=lambda x: x[8], reverse=True)
    x1, y1, x2, y2, cx, cy, area, conf, flower_score = candidates[0]

    # Draw best bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img, f"Flower {flower_score:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )

    return img, [[cx, cy], area, conf, flower_score]


def trackFlower(drone, info, w, h, pid, prev_error):
    error = info[0][0] - w//2
    speed = pid[0]*error + pid[1]*(error - prev_error)
    speed = int(np.clip(speed, -100, 100))

    if info[0][0] != 0:
        drone.yaw_velocity = speed
    else:
        drone.yaw_velocity = 0

    try:
        drone.send_rc_control(0,0,0,drone.yaw_velocity)
    except:
        pass

    return error
