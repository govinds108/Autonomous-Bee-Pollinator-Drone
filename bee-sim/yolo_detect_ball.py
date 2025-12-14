from ultralytics import YOLO
import cv2
from state_mapping import bbox_to_state

IMG_PATH = "frame_ball.png"

model = YOLO("yolov8n.pt")  # COCO pretrained

results = model.predict(source=IMG_PATH, conf=0.20, verbose=False)
r = results[0]

if r.boxes is None or len(r.boxes) == 0:
    raise RuntimeError("No detections — move ball closer or lower conf.")

# pick highest confidence detection
best = max(r.boxes, key=lambda b: float(b.conf[0].item()))

cls_id = int(best.cls[0].item())
name = model.names[cls_id]
conf = float(best.conf[0].item())

x1, y1, x2, y2 = [float(v) for v in best.xyxy[0].tolist()]
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
area = (x2 - x1) * (y2 - y1)

print(f"class: {name}  conf={conf:.2f}")
print(f"center: ({cx:.1f}, {cy:.1f})")
print(f"area: {area:.1f}")

annotated = r.plot()
cv2.imwrite("frame_ball_annotated.png", annotated)
print("Saved frame_ball_annotated.png")

img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]

state, region_id, bin_id, closeness = bbox_to_state(x1, y1, x2, y2, w, h)
print("STATE:", state, "region:", region_id, "bin:", bin_id, "closeness:", closeness)
