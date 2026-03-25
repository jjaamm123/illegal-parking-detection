import cv2
import numpy as np


class ROIManager:

    ZONE_FILL_COLOR   = (0, 0, 200)
    ZONE_BORDER_COLOR = (0, 0, 255)
    ZONE_LABEL_COLOR  = (255, 255, 255)

    def __init__(self):
        self.zones = []
        self._zone_names = []

    def define_zones_interactive(self, frame):
        self.zones = []
        self._zone_names = []
        current_points = []                   
        canvas = frame.copy()

        def _draw_instructions(img):         
            cv2.putText(
                img,
                "L-Click: Add point | R-Click: Close polygon | R: Reset | Q: Finish",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )

        _draw_instructions(canvas)

        def mouse_callback(event, x, y, flags, param):
            nonlocal canvas, current_points      

            if event == cv2.EVENT_LBUTTONDOWN:
                current_points.append((x, y))
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
                if len(current_points) > 1:
                    cv2.line(canvas, current_points[-2], current_points[-1], (0, 255, 0), 2)
                cv2.imshow("Define No-Parking Zones", canvas)

            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(current_points) >= 3:
                    pts = np.array(current_points, np.int32)
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [pts], (0, 0, 180))              
                    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas) 
                    cv2.polylines(canvas, [pts], True, self.ZONE_BORDER_COLOR, 2)

                    zone_name = f"Zone {len(self.zones) + 1}"
                    self.zones.append(current_points.copy())
                    self._zone_names.append(zone_name)

                    cx = int(np.mean([p[0] for p in current_points]))
                    cy = int(np.mean([p[1] for p in current_points]))
                    cv2.putText(canvas, zone_name, (cx - 20, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ZONE_LABEL_COLOR, 2)

                    current_points = []
                    cv2.imshow("Define No-Parking Zones", canvas)
                else:
                    print("[ROI] Need at least 3 points to form a zone.")

        cv2.namedWindow("Define No-Parking Zones", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Define No-Parking Zones", 960, 540)
        cv2.setMouseCallback("Define No-Parking Zones", mouse_callback)
        cv2.imshow("Define No-Parking Zones", canvas)

        while True:
            key = cv2.waitKey(1) & 0xFF            

            if key == ord('q'):
                break

            elif key == ord('r'):                 
                canvas = frame.copy()
                _draw_instructions(canvas)
                current_points = []
                self.zones = []
                self._zone_names = []
                cv2.imshow("Define No-Parking Zones", canvas)

        cv2.destroyWindow("Define No-Parking Zones")
        return self.zones

    def is_inside_zone(self, point, zone):
        pts = np.array(zone, np.int32)
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0

    def get_vehicle_zone(self, centroid):
        for i, zone in enumerate(self.zones):
            if self.is_inside_zone(centroid, zone):
                return i
        return -1

    def draw_zones(self, frame):
        overlay = frame.copy()
        for i, zone in enumerate(self.zones):
            pts = np.array(zone, np.int32)
            cv2.fillPoly(overlay, [pts], self.ZONE_FILL_COLOR)
            cv2.polylines(frame, [pts], True, self.ZONE_BORDER_COLOR, 2)
            cx = int(np.mean([p[0] for p in zone]))
            cy = int(np.mean([p[1] for p in zone]))
            cv2.putText(frame, self._zone_names[i], (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.ZONE_LABEL_COLOR, 2)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        return frame