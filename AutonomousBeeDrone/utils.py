from djitellopy import Tello
import cv2
import numpy as np

last_tracked_label = None
DEBUG_SHOW_MASK = False


def initializeTello():
    """Initialize Tello and start video stream."""
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


def findFlower(img):
    """
    Detect flowers using color segmentation (LAB + HSV).
    Returns:
        img: image with bounding boxes
        [center, area]: (cx, cy), area of the largest detected flower
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Detect bright petals
    petals_mask = cv2.inRange(L, 160, 255)

    # Detect green stems
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Combine masks
    mask = cv2.bitwise_or(petals_mask, green_mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Edge detection
    edges = cv2.Canny(mask, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flowerListC, flowerListArea = [], []

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            x, y, w_box, h_box = cv2.boundingRect(c)
            aspect_ratio = w_box / float(h_box)
            if 0.5 < aspect_ratio < 1.6:
                cx, cy = x + w_box // 2, y + h_box // 2
                flowerListC.append([cx, cy])
                flowerListArea.append(area)
                cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(img, f"flower {area:.0f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
