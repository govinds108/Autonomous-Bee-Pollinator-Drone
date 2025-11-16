from utils import *
import cv2
import numpy as np
from ultralytics import YOLO

# Frame size and drone
w, h = 720, 440
myDrone = initializeTello()
device = "cpu"

print("Loading YOLO-Nano model...")
model = YOLO('yolov8n.pt')
print("YOLO model loaded.")
yolo_confidence = 0.25
flower_score_threshold = 0.4

# Inform user how to operate (manual takeoff)
print("Manual takeoff enabled. Press 't' to takeoff, 'l' to land, 'q' to quit.")

while True:
    # Get drone camera frame
    img = telloGetFrame(myDrone, w, h)

    # Detect flower using YOLO + visual filters
    img, info = detectFlower(img, model, conf_thresh=yolo_confidence,
                              flower_thresh=flower_score_threshold)

    # detectFlower returns: [[cx, cy], area, conf, flower_score]
    (cx, cy), area, conf, flower_score = info[0], info[1], info[2], info[3]

    # Overlay detection info on frame
    cv2.putText(img, f"YOLO conf: {conf:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"Flower score: {flower_score:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Area: {area}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Draw center and line to image center when a detection exists
    if cx != 0 or cy != 0:
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.line(img, (w // 2, h // 2), (int(cx), int(cy)), (255, 0, 0), 1)

    # Display feed
    cv2.imshow("Tello Flower Detection", img)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        try:
            myDrone.land()
        except Exception:
            pass
        break
    elif key == ord('t'):
        # Manual takeoff with battery check and exception handling
        try:
            batt = None
            try:
                batt = myDrone.get_battery()
            except Exception:
                pass
            if batt is not None and batt < 20:
                print(f"Battery low ({batt}%). Aborting takeoff. Charge drone before flying.")
            else:
                try:
                    myDrone.takeoff()
                except Exception as e:
                    print("Takeoff failed:", e)
        except Exception:
            # catch-all to avoid loop exit on unexpected errors
            print("Takeoff command raised an error.")
    elif key == ord('l'):
        try:
            myDrone.land()
        except Exception as e:
            print("Land failed:", e)
    elif key == ord('c'):
        cv2.imwrite('captured_frame.jpg', img)
        print("Saved frame to captured_frame.jpg")
