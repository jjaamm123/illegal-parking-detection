import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:

    VEHICLE_CLASS_IDS = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence_threshold=0.20,
        use_background_subtraction=True,
        yolo_input_size=480,
        nms_iou_threshold=0.45,
    ):
        print(f"[Detector] Loading model: {model_path}")
        self.model                      = YOLO(model_path)
        self.confidence_threshold       = confidence_threshold
        self.use_background_subtraction = use_background_subtraction
        self.yolo_input_size            = yolo_input_size
        self.nms_iou_threshold          = nms_iou_threshold

        if use_background_subtraction:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=80,
                detectShadows=True,
            )

        print("[Detector] Ready.")

    def detect(self, frame):
        yolo_dets = self._detect_yolo(frame)

        if self.use_background_subtraction:
            bg_dets  = self._detect_background_subtraction(frame)
            combined = self._merge_detections(yolo_dets, bg_dets)
        else:
            combined = yolo_dets

        return self._apply_nms(combined)

    # ── YOLO ─────────────────────────────────────────────────────────────────

    def _detect_yolo(self, frame):
        h, w  = frame.shape[:2]
        scale = self.yolo_input_size / max(h, w)

        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        scale = scale if scale < 1.0 else 1.0

        results    = self.model(small, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.VEHICLE_CLASS_IDS:
                continue

            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = (
                int(x1 / scale), int(y1 / scale),
                int(x2 / scale), int(y2 / scale),
            )

            detections.append((x1, y1, x2, y2, self.VEHICLE_CLASS_IDS[cls_id], conf))

        return detections

    # ── MOG2 background subtraction ───────────────────────────────────────────

    def _detect_background_subtraction(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w       = frame.shape[:2]
        frame_area = h * w
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < frame_area * 0.0015 or area > frame_area * 0.06:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / (bh + 1e-6)

            if aspect < 0.5 or aspect > 3.5:
                continue

            if bw < 30 or bh < 20:
                continue

            # Classify the blob by its pixel area instead of defaulting everything to "car"
            # Thresholds are relative to frame size so they work across resolutions
            label = self._classify_by_size(area, frame_area)
            detections.append((x, y, x + bw, y + bh, label, 0.50))

        return detections

    def _classify_by_size(self, blob_area, frame_area):
        """
        Estimate vehicle type from the blob's pixel area relative to the frame.
        Top-down vehicles have predictable relative sizes:
            motorcycle — very small footprint
            car        — standard footprint
            truck/bus  — large footprint
        These thresholds work for typical 720p–1080p aerial footage.
        """
        ratio = blob_area / frame_area

        if ratio < 0.003:
            return "motorcycle"
        elif ratio < 0.012:
            return "car"
        elif ratio < 0.030:
            return "truck"
        else:
            return "bus"

    # ── Merge YOLO + MOG2 ─────────────────────────────────────────────────────

    def _merge_detections(self, yolo_dets, bg_dets):
        if not bg_dets:
            return yolo_dets
        if not yolo_dets:
            return bg_dets

        merged = list(yolo_dets)

        for bg_det in bg_dets:
            bx1, by1, bx2, by2, _, _ = bg_det
            overlaps = False
            for y_det in yolo_dets:
                yx1, yy1, yx2, yy2, _, _ = y_det
                if self._iou((bx1, by1, bx2, by2), (yx1, yy1, yx2, yy2)) > 0.25:
                    overlaps = True
                    break
            if not overlaps:
                merged.append(bg_det)

        return merged

    # ── NMS across all detections ─────────────────────────────────────────────

    def _apply_nms(self, detections):
        if not detections:
            return []

        boxes  = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
        scores = np.array([d[5] for d in detections],                      dtype=np.float32)

        boxes_xywh       = boxes.copy()
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_iou_threshold,
        )

        if len(indices) == 0:
            return []

        return [detections[int(i)] for i in indices]

    # ── IoU helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _iou(boxA, boxB):
        xA    = max(boxA[0], boxB[0])
        yA    = max(boxA[1], boxB[1])
        xB    = min(boxA[2], boxB[2])
        yB    = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)