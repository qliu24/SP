import numpy as np
import math
from BoundingBox import BoundingBox


class DenseNMS:
    def __init__(self, stride, threshold, box_length=100):
        self.stride = stride
        self.threshold = threshold
        self.half_box_length = box_length / 2
        self.mask = self.get_elon_musk()  # 2d mask with shape (radius + 1 + radius, radius + 1 + radius)
        self.radius = self.mask.shape[0] // 2

    def iou(self, x1, y1, x2, y2):
        bounding_box1 = BoundingBox(x1 * self.stride - self.half_box_length, x1 * self.stride + self.half_box_length,
                                    y1 * self.stride - self.half_box_length, y1 * self.stride + self.half_box_length)
        bounding_box2 = BoundingBox(x2 * self.stride - self.half_box_length, x2 * self.stride + self.half_box_length,
                                    y2 * self.stride - self.half_box_length, y2 * self.stride + self.half_box_length)
        assert(bounding_box1.iou(bounding_box2) == bounding_box2.iou(bounding_box1))
        return bounding_box1.iou(bounding_box2)

    def near(self, x1, y1, x2, y2):
        return self.iou(x1, y1, x2, y2) > self.threshold

    def get_elon_musk(self):
        # TODO: radius could be smaller but seems not necessary
        radius = int(math.ceil(self.half_box_length * 2 / self.stride))
        mask = np.ones((radius + 1 + radius, radius + 1 + radius))
        # center at (radius, radius)
        for i in range(0, radius + 1):
            for j in range(0, radius + 1):
                if self.near(0, 0, i, j):
                    mask[radius + i, radius + j] = 0
                    mask[radius + i, radius - j] = 0
                    mask[radius - i, radius - j] = 0
                    mask[radius - i, radius + j] = 0
        return mask

    def nms(self, score_map):
        padded = np.pad(score_map - np.amin(score_map) + 1, self.radius, 'constant', constant_values=0)
        max_arg = np.unravel_index(np.argmax(padded), padded.shape)
        return_list = []
        while padded[max_arg] > 0.5:
            return_list.append((max_arg[0] - self.radius, max_arg[1] - self.radius))
            padded[max_arg[0] - self.radius: max_arg[0] + self.radius + 1, max_arg[1] - self.radius: max_arg[1] + self.radius + 1] *= self.mask
            max_arg = np.unravel_index(np.argmax(padded), padded.shape)
        return return_list
