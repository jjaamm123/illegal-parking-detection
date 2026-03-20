import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
model = YOLO("yolov8n.pt")

def detect_vehicles(frame):
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:          
            continue

        conf = float(box.conf[0])                  
        if conf < 0.40:                            
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = VEHICLE_CLASSES[cls_id] 
        detections.append((x1, y1, x2, y2, cls_id))

    return detections
