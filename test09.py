import numpy as np

bbox =np.array([[24, 50, 48, 120],
       [20, 30, 64, 100]])
bbox = np.stack(bbox)
_bbox = bbox.copy()

h = bbox[:, 3] - bbox[:, 1]
w = bbox[:, 2] - bbox[:, 0]
max_side = np.maximum(h, w)

_bbox[:, 0] = bbox[:, 0] + w / 2 - max_side / 2
_bbox[:, 1] = bbox[:, 1] + h / 2 - max_side / 2
_bbox[:, 2] = _bbox[:, 0] + max_side
_bbox[:, 3] = _bbox[:, 1] + max_side
print(_bbox)


