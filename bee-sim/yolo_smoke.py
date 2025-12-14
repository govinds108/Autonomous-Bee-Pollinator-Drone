from ultralytics import YOLO

model = YOLO("yolov8n.pt")
r = model.predict("frame_ball.png", conf=0.2, verbose=False)[0]

print("num boxes:", 0 if r.boxes is None else len(r.boxes))
if r.boxes is not None and len(r.boxes):
    b = max(r.boxes, key=lambda x: float(x.conf[0]))
    print("class_id:", int(b.cls[0]), "conf:", float(b.conf[0]))
    print("xyxy:", [float(v) for v in b.xyxy[0].tolist()])
