import torch
import numpy as np

a = np.array([[1,1,1,1,3],
              [1,1,1,1,1],
              [1,1,1,1,2]])

order = a[:, 4].argsort()[::-1]
print(order)
c = a[order]
print(c[0])
print(c[1:])



