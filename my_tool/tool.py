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
        ovr = np.true_divide(inter_area, (box_area + boxes_area -inter_area))
    return ovr


def NMS(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    order = boxes[:, 4].argsort()[::-1]
    boxes_ = boxes[order]
    r_boxes = []

    while boxes_.shape[0] > 0:
        max_box = boxes_[0]
        _boxes = boxes_[1:]
        r_boxes.append(max_box)
        if _boxes.shape[0] == 0:
            break
        index = np.where(iou(max_box, _boxes, isMin) < thresh)
        boxes_ = _boxes[index]
    return np.stack(r_boxes)


def convert_to_square(bbox):
    _bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])

    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)

    _bbox[:, 0] = bbox[:, 0] + w / 2 - max_side / 2
    _bbox[:, 1] = bbox[:, 1] + h / 2 - max_side / 2
    _bbox[:, 2] = _bbox[:, 0] + max_side
    _bbox[:, 3] = _bbox[:, 1] + max_side
    return _bbox
