from utils import *
import cv2

w, h = 720, 440
pid = [0.4, 0.4, 0]  # Adjusted PID values for smoother flower tracking
myDrone = initializeTello()
pError = 0
startCounter = 1  # No flight set to 1

while True:
    ## Flight
    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1

    ## Step 1: Get drone camera frame
    img = telloGetFrame(myDrone, w, h)

    ## Step 2: Detect flowers
    img, info = findFlower(img)

    ## Step 3: Track detected flower
    pError = trackFlower(myDrone, info, w, h, pid, pError)

    ## Display feed with flower detection
    cv2.imshow("Tello Flower Detection", img)

    ## Controls: 
    ## 'q' to quit and land
    ## 't' to takeoff
    ## 'l' to land
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        myDrone.land()
        break
    elif key == ord('t'):
        myDrone.takeoff()
    elif key == ord('l'):
        myDrone.land()
    elif key == ord('c'):
        # Capture and save current frame for offline testing
        save_path = 'captured_frame.jpg'
        cv2.imwrite(save_path, img)
        print(f"Saved frame to {save_path}")