import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader


class MyDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path, "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(self.path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(self.path, "part.txt")).readlines())

    def __getitem__(self, index):
        print(type(self.dataset[0]))
        strs = self.dataset[index].strip().split(" ")
        strs = list(filter(lambda x: bool(x), strs))
        #  print(strs[0])
        img_path = os.path.join(self.path, strs[0])
        cond = torch.Tensor([int(strs[1])])
        offset = torch.Tensor(list(map(float, strs[2:])))
        im_data = (np.array(Image.open(img_path)) / 255. - 0.5).transpose([2, 0, 1])
        img_data = torch.Tensor(im_data)
        return img_data, cond, offset

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    data = MyDataSet(r"E:\celeba\sample\12")
    data.__getitem__(0)
    # secret = DataLoader(data, batch_size=64, shuffle=True, drop_last=True)
    # for i, (img_data, cond, offset) in enumerate(secret):
    #     print(i, img_data.size())
