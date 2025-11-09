from djitellopy import Tello
import cv2
import numpy as np

# Global YOLO model and last detected label
model = None
last_tracked_label = None

# Toggle this to True to see *all* detections every frame
FORCE_PRINT_DETECTIONS = True

# If True, filters only "potted plant" detections
FILTER_PLANTS_ONLY = True


def initializeTello():
    """Initialize Tello and start video stream."""
    global model
    model = None

    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0

    try:
        print(f"Battery Life Percentage: {myDrone.get_battery()}%")
    except Exception:
        print("Could not read battery level")

    myDrone.streamoff()
    myDrone.streamon()
    return myDrone


def telloGetFrame(myDrone, w=360, h=240):
    """Retrieve and resize current Tello video frame."""
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img


def get_model():
    """Lazily load YOLOv8 nano model (COCO-trained)."""
    global model
    if model is not None:
        return model
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8n model loaded successfully (COCO 80 classes).")
        return model
    except Exception as e:
        print(f"❌ YOLO load failed: {e}")
        model = None
        return None

def findFlower(img):
    """
    Detect flowers using color segmentation (LAB + HSV).
    Returns:
        img: image with drawn bounding boxes
        [center, area]: (cx, cy), area of the largest detected flower
    """

    # Convert to multiple color spaces for robust color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 1️⃣ Detect bright/light regions (flower petals)
    petals_mask = cv2.inRange(L, 160, 255)

    # 2️⃣ Detect greenish tones (stems, leaves)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 3️⃣ Combine masks
    mask = cv2.bitwise_or(petals_mask, green_mask)

    # 4️⃣ Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5️⃣ Edge detection
    edges = cv2.Canny(mask, 50, 150)

    # 6️⃣ Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flowerListC, flowerListArea = [], []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:  # ignore small blobs
            x, y, w_box, h_box = cv2.boundingRect(c)
            aspect_ratio = w_box / float(h_box)
            # keep roughly square-ish shapes
            if 0.5 < aspect_ratio < 1.6:
                cx, cy = x + w_box // 2, y + h_box // 2
                flowerListC.append([cx, cy])
                flowerListArea.append(area)
                cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(img, f"flower {area:.0f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7️⃣ Select the largest detected flower
    global last_tracked_label
    if len(flowerListArea) > 0:
        i = np.argmax(flowerListArea)
        if last_tracked_label != "flower":
            print(f"Tracking object: flower area={flowerListArea[i]:.1f}")
            last_tracked_label = "flower"
        return img, [flowerListC[i], flowerListArea[i]]
    else:
        if last_tracked_label is not None:
            print("Tracking object lost")
            last_tracked_label = None
        return img, [[0, 0], 0]
    
def trackFlower(myDrone, info, w, h, pid, pError):
    """PID-based movement control to follow detected potted plant."""
    error = info[0][0] - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    vertical_error = info[0][1] - h // 2
    up_down_speed = pid[0] * vertical_error
    up_down_speed = int(np.clip(up_down_speed, -50, 50))

    forward_speed = 0
    if info[1] > 0:
        if info[1] < 20000:
            forward_speed = 20
        elif info[1] > 40000:
            forward_speed = -20

    # print(f"Yaw Speed: {speed}, Up-Down: {up_down_speed}, Forward: {forward_speed}")

    if info[0][0] != 0:
        myDrone.yaw_velocity = speed
        myDrone.up_down_velocity = up_down_speed
        myDrone.for_back_velocity = forward_speed
    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 20
        error = 0

    if myDrone.send_rc_control:
        myDrone.send_rc_control(
            myDrone.left_right_velocity,
            myDrone.for_back_velocity,
            myDrone.up_down_velocity,
            myDrone.yaw_velocity,
        )

    return error
