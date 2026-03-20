import cv2
from detector import detect_vehicles
from tracker import CentroidTracker
from roi import ROIManager                         
from timer_check import (
    vehicle_entered, vehicle_exited,
    get_duration, is_illegal,
    record_violation, get_all_violations          
)
from visualizer import draw_vehicle               

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, first_frame = cap.read()

roi_manager = ROIManager()
zones = roi_manager.define_zones_interactive(first_frame)

tracker = CentroidTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_vehicles(frame)
    rects = [(d[0], d[1], d[2], d[3]) for d in detections]
    objects = tracker.update(rects)

    roi_manager.draw_zones(frame)             

    for vehicle_id, centroid in objects.items():
        cx, cy   = int(centroid[0]), int(centroid[1])
        zone_idx = roi_manager.get_vehicle_zone((cx, cy)) 
        in_zone  = zone_idx >= 0

        for det in detections:
            x1, y1, x2, y2, vtype = det[0], det[1], det[2], det[3], det[4]
            duration     = 0.0
            is_violation = False

            if in_zone:
                zone_name = f"Zone {zone_idx + 1}"
                vehicle_entered(vehicle_id, zone_name)
                duration     = get_duration(vehicle_id)
                is_violation = is_illegal(vehicle_id)

                if is_violation:                
                    record_violation(vehicle_id, zone_name, vtype)
            else:
                vehicle_exited(vehicle_id)

            draw_vehicle(frame, (x1, y1, x2, y2),
                        vehicle_id, vtype,
                        in_zone, is_violation, duration)


    total = len(get_all_violations())
    cv2.putText(frame, f"Violations: {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

    cv2.imshow("Illegal Parking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()