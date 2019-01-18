import numpy as np


def iou(box, boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2-xx1)
    h = np.maximum(0, yy2-yy1)

    inter_area = w*h

    iou_value = inter_area / (box_area + boxes_area - inter_area)

    return iou_value