import numpy as np


def iou(box, boxes, isMin = False):
    top_x = np.maximum(box[0], boxes[:, 0])
    top_y = np.maximum(box[1], boxes[:, 1])
    bottom_x = np.minimum(box[2], boxes[:, 2])
    bottom_y = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, bottom_x - top_x)
    h = np.maximum(0, bottom_y - top_y)

    inter_area = w * h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    if isMin:
        ovr = np.true_divide(inter_area, np.minimum(box_area, boxes_area))
    else:
        ovr = np.true_divide(inter_area, (boxes_area + box_area - inter_area))

    return ovr
