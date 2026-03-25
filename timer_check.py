import time
import datetime
class ParkingTimer:                               

    def __init__(self, threshold_seconds=30):     
        self.threshold    = threshold_seconds
        self._entry_times = {}
        self._zone_map    = {}                    
        self.violations   = {}                    

    def update_threshold(self, threshold_seconds): 
        self.threshold = threshold_seconds

    def vehicle_in_zone(self, vehicle_id, zone_index, zone_name): 
        if vehicle_id not in self._entry_times:
            self._entry_times[vehicle_id] = time.time()
            self._zone_map[vehicle_id] = zone_index           

    def vehicle_out_of_zone(self, vehicle_id):    
        self._entry_times.pop(vehicle_id, None)
        self._zone_map.pop(vehicle_id, None)       

    def get_duration(self, vehicle_id):
        if vehicle_id in self._entry_times:
            return time.time() - self._entry_times[vehicle_id]
        return 0.0

    def is_violation(self, vehicle_id):          
        return self.get_duration(vehicle_id) >= self.threshold

    def record_violation(self, vehicle_id, zone_name, vehicle_type, timestamp=None):
        duration = self.get_duration(vehicle_id)
        ts = timestamp or datetime.datetime.now().strftime("%H:%M:%S")

        if vehicle_id not in self.violations:
            self.violations[vehicle_id] = {
                "vehicle_id":   vehicle_id,
                "vehicle_type": vehicle_type,
                "zone":         zone_name,
                "timestamp":    ts,
                "duration":     duration,
            }
            return True                           
        else:
            self.violations[vehicle_id]["duration"] = duration 
            return False                        

    def get_all_violations(self):
        return list(self.violations.values())

    def clear(self):                              
        self._entry_times.clear()
        self._zone_map.clear()
        self.violations.clear()