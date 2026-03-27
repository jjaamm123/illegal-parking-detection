from ultralytics import YOLO
class VehicleDetector:                            

    VEHICLE_CLASS_IDS = {                          
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.40):
        print(f"[Detector] Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold  
        print("[Detector] Model loaded successfully.")

    def detect(self, frame):                       
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.VEHICLE_CLASS_IDS:       
                continue
            conf = float(box.conf[0]) 
            if conf < self.confidence_threshold:          
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.VEHICLE_CLASS_IDS[cls_id]  
            detections.append((x1, y1, x2, y2, class_name, conf))  

        return detections