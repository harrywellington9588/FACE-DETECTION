from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch


class face_dataset(Dataset):
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path, str(self.size), "positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(self.path, str(self.size), "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(self.path, str(self.size), "part.txt")).readlines())

    def __getitem__(self, index):
        lines = self.dataset[0]
        line = lines.strip().split(" ")
        line = list(filter(lambda x: bool(x), line))
        image_name = line[0]
        img_path = os.path.join(self.path, str(self.size), image_name)
        image = np.array(Image.open(img_path))
        img = torch.Tensor((image / 255 - 0.5).transpose([2, 0, 1]))

        # transform = transforms.Compose(
        #     transforms.ToTensor()
        # )
        # img = transform(image)
        cond = torch.Tensor([int(line[-1])])
        offset = torch.Tensor(list(map(float, line[1:-1])))

        return img, cond, offset

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    path = r"E:\celeba\train_dataset"
    img_size = 12
    data = face_dataset(path, img_size)
    img, cond, offset = data.__getitem__(0)

    print(img.shape)
    print(cond)
    print(offset)


