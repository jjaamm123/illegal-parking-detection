class CentroidTracker:

    def __init__(self):
        self.next_object_id = 0
        self.objects = {}

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.next_object_id += 1

    def update(self, rects):
        if len(rects) == 0:
            return self.objects

        for rect in rects:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.register([cx, cy])

        return self.objects