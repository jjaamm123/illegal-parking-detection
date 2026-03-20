import cv2
import numpy as np

class ROIManager:                                 

    def __init__(self):
        self.zones = []                            
        self._zone_names = []                      
        self.current_points = []

    def define_zones_interactive(self, frame):
        canvas = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_points.append((x, y))
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)  
                cv2.imshow("Define Zones", canvas)

            elif event == cv2.EVENT_RBUTTONDOWN:           
                if len(self.current_points) >= 3:
                    pts = np.array(self.current_points, np.int32)
                    cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)  
                    zone_name = f"Zone {len(self.zones) + 1}"          
                    self.zones.append(self.current_points.copy())
                    self._zone_names.append(zone_name)                  
                    self.current_points = []
                    cv2.imshow("Define Zones", canvas)

        cv2.namedWindow("Define Zones")
        cv2.setMouseCallback("Define Zones", mouse_callback)
        cv2.imshow("Define Zones", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.zones

    def is_inside_zone(self, point, zone):         
        pts = np.array(zone, np.int32)
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0