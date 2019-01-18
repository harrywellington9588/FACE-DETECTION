import numpy as np
import torch

a = np.array([1,2,3,5,9,6,1])

index = np.where(a < 5)
print(index)
b = a[index]
print(b)

x = np.random.randint(0, 2, [1, 3, 12, 12])
x = torch.Tensor(x)
print(type(x))