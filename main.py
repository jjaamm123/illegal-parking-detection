
import cv2
from detector import detect_vehicles
from tracker import CentroidTracker
from roi import define_zone, zones, is_inside_zone
from timer_check import vehicle_entered, vehicle_exited, get_duration, is_illegal
from visualizer import draw_box

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, first_frame = cap.read()
if ret:
    define_zone(first_frame)

tracker = CentroidTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break


    detections = detect_vehicles(frame)
    rects = [(d[0], d[1], d[2], d[3]) for d in detections]


    objects = tracker.update(rects)


    for vehicle_id, centroid in objects.items():
        cx, cy = int(centroid[0]), int(centroid[1])
        in_zone = any(is_inside_zone((cx, cy), z) for z in zones)

        if in_zone:
            vehicle_entered(vehicle_id)
            duration = get_duration(vehicle_id)
            illegal  = is_illegal(vehicle_id)
            label    = f"ID:{vehicle_id} ILLEGAL({duration:.0f}s)" if illegal else f"ID:{vehicle_id} In Zone"
        else:
            vehicle_exited(vehicle_id)
            label = f"ID:{vehicle_id}"


        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            draw_box(frame, (x1, y1, x2, y2), label)

    cv2.imshow("Illegal Parking Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()