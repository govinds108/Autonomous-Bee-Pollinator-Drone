from ultralytics import YOLO
import sys
import json

img_path = sys.argv[1] if len(sys.argv) > 1 else "step.png"

model = YOLO("yolov8n.pt")
r = model.predict(img_path, conf=0.2, verbose=False)[0]

if r.boxes is None or len(r.boxes) == 0:
    print(json.dumps({"ok": False}))
    sys.exit(0)

# choose best "sports ball" if present else best overall
best = None
best_conf = -1.0
for b in r.boxes:
    cls_id = int(b.cls[0])
    name = model.names.get(cls_id, str(cls_id))
    conf = float(b.conf[0])
    if name == "sports ball" and conf > best_conf:
        best, best_conf = b, conf

if best is None:
    best = max(r.boxes, key=lambda bb: float(bb.conf[0]))

x1, y1, x2, y2 = [float(v) for v in best.xyxy[0].tolist()]
cls_id = int(best.cls[0])
name = model.names.get(cls_id, str(cls_id))
conf = float(best.conf[0])

print(json.dumps({
    "ok": True,
    "name": name,
    "conf": conf,
    "x1": x1, "y1": y1, "x2": x2, "y2": y2
}))
