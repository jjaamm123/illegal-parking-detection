import cv2


class Visualizer:                                  

    COLOR_NORMAL    = (50, 200, 50)
    COLOR_WARNING   = (30, 150, 255)
    COLOR_VIOLATION = (30, 30, 220)
    FONT            = cv2.FONT_HERSHEY_SIMPLEX

    def draw_vehicle(self, frame, bbox, vehicle_id, vehicle_type, duration, is_violation, in_zone):
        x1, y1, x2, y2 = bbox

        if is_violation:
            color  = self.COLOR_VIOLATION
            status = f"!! ILLEGAL PARKING ({duration:.0f}s)"
        elif in_zone:
            color  = self.COLOR_WARNING
            status = f"In Zone: {duration:.0f}s"
        else:
            color  = self.COLOR_NORMAL
            status = vehicle_type.upper()

        label = f"ID:{vehicle_id}  {status}"
        self._draw_box_with_label(frame, x1, y1, x2, y2, label, color)  
        if is_violation:                           
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLOR_VIOLATION, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        return frame

    def _draw_box_with_label(self, frame, x1, y1, x2, y2, label, color):  
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, self.FONT, 0.48, 1)
        label_y1 = max(y1 - text_h - 8, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 3),
                    self.FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_alert_banner(self, frame, active_violation_count):    
        if active_violation_count > 0:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (20, 20, 180), -1)
            text = f"  ALERT: {active_violation_count} ILLEGAL PARKING VIOLATION(S) DETECTED"
            cv2.putText(frame, text, (8, 34),
                        self.FONT, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_info_panel(self, frame, total_detected, total_violations, fps): 
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 30), -1)
        info = (f"  Vehicles: {total_detected}   |   "
                f"Violations: {total_violations}   |   "
                f"FPS: {fps:.1f}")
        cv2.putText(frame, info, (8, h - 10),
                    self.FONT, 0.58, (200, 200, 210), 1, cv2.LINE_AA)
        return frame