from ultralytics import YOLO
import cv2

IMG_PATH = "frame0.png"

# Small, CPU-friendly model
model = YOLO("yolov8n.pt")  # downloads weights once

results = model.predict(source=IMG_PATH, conf=0.25, verbose=False)

r = results[0]
img = cv2.imread(IMG_PATH)

if r.boxes is None or len(r.boxes) == 0:
    print("No detections found on frame0.png")
else:
    print(f"Detections: {len(r.boxes)}")
    for i, b in enumerate(r.boxes):
        cls_id = int(b.cls[0].item())
        name = model.names[cls_id]
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        print(f"{i}: {name} conf={conf:.2f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) "
              f"center=({cx:.1f},{cy:.1f}) area={area:.1f}")

# Save annotated output
annotated = r.plot()  # returns an annotated image (numpy array)
cv2.imwrite("frame0_annotated.png", annotated)
print("Saved frame0_annotated.png")
