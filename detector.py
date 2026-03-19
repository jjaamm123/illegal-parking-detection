from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_vehicles(frame):
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2, cls_id))

    return detections