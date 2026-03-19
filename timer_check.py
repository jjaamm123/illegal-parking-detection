import time

entry_times = {}

def vehicle_entered(vehicle_id):
    if vehicle_id not in entry_times:
        entry_times[vehicle_id] = time.time()

def get_duration(vehicle_id):
    if vehicle_id in entry_times:
        return time.time() - entry_times[vehicle_id]
    return 0.0

def is_illegal(vehicle_id, threshold=30):
    return get_duration(vehicle_id) >= threshold