import cv2
import time
import datetime
import os

from detector    import VehicleDetector
from tracker     import CentroidTracker
from roi         import ROIManager
from timer_check import ParkingTimer
from visualizer  import Visualizer


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def run_detection(
    video_path,
    zones,
    threshold,
    output_path=None,
    frame_callback=None,
    log_callback=None,
    stop_flag=None,
    stats_callback=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open '{video_path}'")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))
        print(f"[Main] Saving output to: {output_path}")

    detector = VehicleDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.25,
        use_background_subtraction=True,
        yolo_input_size=480,
        nms_iou_threshold=0.45,
    )
    tracker       = CentroidTracker(max_disappeared=50)
    roi_manager   = ROIManager()
    parking_timer = ParkingTimer(threshold_seconds=threshold)
    visualizer    = Visualizer()

    roi_manager.zones       = zones
    roi_manager._zone_names = [f"Zone {i + 1}" for i in range(len(zones))]

    vehicle_type_map = {}
    vehicle_bbox_map = {}

    frame_count   = 0
    start_time    = time.time()
    fps_display   = 0.0
    WARMUP_FRAMES = 40
    FRAME_SKIP    = 3

    last_objects = {}

    print(f"[Main] Warming up ({WARMUP_FRAMES} frames)...")
    print("[Main] Detection started.")

    while True:
        if stop_flag and stop_flag[0]:
            print("[Main] Stopped.")
            break

        ret, frame = cap.read()
        if not ret:
            print("[Main] End of video.")
            break

        frame_count += 1

        if frame_count % 15 == 0:
            elapsed     = time.time() - start_time
            fps_display = frame_count / elapsed if elapsed > 0 else 0.0

        # Always feed bg subtractor to keep background model current
        if detector.use_background_subtraction:
            detector.bg_subtractor.apply(frame)

        if frame_count <= WARMUP_FRAMES:
            warmup_frame = frame.copy()
            pct = int((frame_count / WARMUP_FRAMES) * 100)
            cv2.rectangle(warmup_frame, (0, 0), (width, 44), (30, 30, 50), -1)
            cv2.putText(warmup_frame,
                        f"  Initialising background model... {pct}%",
                        (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)
            if frame_callback:
                frame_callback(warmup_frame)
            if writer:
                writer.write(warmup_frame)
            continue

        run_full = (frame_count % FRAME_SKIP == 0)

        if run_full:
            detections = detector.detect(frame)
            rects      = [(d[0], d[1], d[2], d[3]) for d in detections]
            objects    = tracker.update(rects)

            for vid, centroid in objects.items():
                cx, cy    = int(centroid[0]), int(centroid[1])
                best_dist = float("inf")
                best_det  = None

                for det in detections:
                    x1, y1, x2, y2, vtype, _ = det
                    dcx  = (x1 + x2) // 2
                    dcy  = (y1 + y2) // 2
                    dist = ((dcx - cx) ** 2 + (dcy - cy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_det  = det

                if best_det and best_dist < 100:
                    x1, y1, x2, y2, vtype, _ = best_det
                    vehicle_type_map[vid] = vtype
                    vehicle_bbox_map[vid] = (x1, y1, x2, y2)

            last_objects = objects

        else:
            objects = last_objects

        roi_manager.draw_zones(frame)

        active_violations = 0

        for vid, centroid in objects.items():
            cx, cy   = int(centroid[0]), int(centroid[1])
            zone_idx = roi_manager.get_vehicle_zone((cx, cy))
            vtype    = vehicle_type_map.get(vid, "vehicle")
            bbox     = vehicle_bbox_map.get(vid)

            in_zone  = zone_idx >= 0
            is_viol  = False
            duration = 0.0

            if in_zone:
                zone_name = roi_manager._zone_names[zone_idx]
                parking_timer.vehicle_in_zone(vid, zone_idx, zone_name)
                duration  = parking_timer.get_duration(vid)
                is_viol   = parking_timer.is_violation(vid)

                if is_viol:
                    active_violations += 1
                    ts        = datetime.datetime.now().strftime("%H:%M:%S")
                    new_entry = parking_timer.record_violation(vid, zone_name, vtype, ts)
                    if new_entry and log_callback:
                        log_callback(parking_timer.violations[vid].copy())
            else:
                parking_timer.vehicle_out_of_zone(vid)

            if bbox:
                visualizer.draw_vehicle(
                    frame, bbox, vid, vtype, duration, is_viol, in_zone
                )

        total_violations = len(parking_timer.violations)
        visualizer.draw_alert_banner(frame, active_violations)
        visualizer.draw_info_panel(frame, len(objects), total_violations, fps_display)

        if stats_callback:
            stats_callback({
                "total_detected":   len(objects),
                "total_violations": total_violations,
                "active_zones":     len(zones),
            })

        if frame_callback:
            frame_callback(frame)

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
        print(f"[Main] Output saved: {output_path}")

    violations = parking_timer.get_all_violations()
    print(f"[Main] Done. {len(violations)} violation(s) recorded.")
    return violations


if __name__ == "__main__":
    from dashboard import launch_dashboard
    launch_dashboard()