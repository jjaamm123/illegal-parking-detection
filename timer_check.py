import time
import datetime                                    

entry_times = {}
violations = {}                                   

def vehicle_entered(vehicle_id, zone_name):       
    if vehicle_id not in entry_times:
        entry_times[vehicle_id] = time.time()

def vehicle_exited(vehicle_id):                 
    entry_times.pop(vehicle_id, None)

def get_duration(vehicle_id):
    if vehicle_id in entry_times:
        return time.time() - entry_times[vehicle_id]
    return 0.0

def is_illegal(vehicle_id, threshold=30):
    return get_duration(vehicle_id) >= threshold

def record_violation(vehicle_id, zone_name, vehicle_type):   
    if vehicle_id not in violations:
        violations[vehicle_id] = {
            "vehicle_id":   vehicle_id,
            "zone":         zone_name,
            "vehicle_type": vehicle_type,
            "timestamp":    datetime.datetime.now().strftime("%H:%M:%S"), 
            "duration":     get_duration(vehicle_id),
        }

def get_all_violations():                         
    return list(violations.values())