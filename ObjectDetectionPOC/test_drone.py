from djitellopy import Tello

tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamoff()  # reset any old streams safely
tello.streamon()   # turn camera stream on again
print("Stream restarted successfully")

tello.end()
