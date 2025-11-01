from utils import *
import cv2

w, h = 720, 440
pid = [0.5, 0.5, 0]
myDrone = initializeTello()
pError = 0
startCounter = 0 # No flight set to 1

while True:

    ## Flight
    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1


    ## Step 1
    img = telloGetFrame(myDrone, w, h)

    ## Step 2
    img, info = findFace(img)
    # print(info[0][0])

    ## Step 3
    pError = trackFace(myDrone, info, w, h, pid, pError)

    cv2.imshow("Tello Camera Feed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break