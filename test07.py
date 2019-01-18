import numpy as np

box = [0.5, 0.5, 0.5, 0.5]

boxes = np.random.rand(10, 4)

top_x = np.maximum(box[0], boxes[:, 0])
print(top_x)
top_y = np.maximum(box[1], boxes[:, 1])
bottom_x = np.minimum(box[2], boxes[:, 2])
bottom_y = np.minimum(box[3], boxes[:, 3])

w = np.maximum(0, bottom_x - top_x)
print(w)
h = np.maximum(0, bottom_y - top_y)

inter_area = w * h
print(inter_area)
box_area = (box[2] - box[0]) * (box[3] - box[1])
boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
ovr = inter_area / (box_area + boxes_area - inter_area)
print(ovr)