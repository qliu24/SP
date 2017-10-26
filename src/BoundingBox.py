class BoundingBox:
    def __init__(self, x1, x2, y1, y2):
        assert x2 >= x1
        assert y2 >= y1
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def intersection_area(self, bounding_box):
        x1 = max(self.x1, bounding_box.x1)
        x2 = min(self.x2, bounding_box.x2)
        y1 = max(self.y1, bounding_box.y1)
        y2 = min(self.y2, bounding_box.y2)
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def union_area(self, bounding_box):
        return self.area() + bounding_box.area() - self.intersection_area(bounding_box)

    def iou(self, bounding_box):
        intersection = self.intersection_area(bounding_box)
        union = self.union_area(bounding_box)
        assert(union > 0)
        return intersection / union

    def area(self):  # must be >= 0
        return (self.x2 - self.x1) * (self.y2 - self.y1)
