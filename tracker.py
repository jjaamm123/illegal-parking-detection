from collections import OrderedDict            
import numpy as np
class CentroidTracker:

    def __init__(self):
        self.next_object_id = 0
        self.objects = {}
        from collections import OrderedDict            
        import numpy as np

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):              
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):        
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1           #
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        for rect in rects:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.register([cx, cy])

        return self.objects