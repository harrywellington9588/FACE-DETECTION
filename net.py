from torch import nn
import torch
import numpy as np
import torch


class pnet(nn.Module):
    def __init__(self):
        super(pnet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 16, 3),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3),
            nn.PReLU(),
        )
        self.conv4_1 = nn.Conv2d(32, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        x = self.pre_layer(x)
        cond = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset


class rnet(nn.Module):
    def __init__(self):
        super(rnet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(28, 48, 3),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(48, 64, 2),
            nn.PReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*2*2, 128),
            nn.PReLU()
        )
        self.fc2_1 = nn.Linear(128, 1)
        self.fc2_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        cond = torch.sigmoid(self.fc2_1(x))
        offset = self.fc2_2(x)
        return cond, offset


class onet(nn.Module):
    def __init__(self):
        super(onet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(32, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 2),
            nn.PReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*2*2, 256),
            nn.PReLU()
        )
        self.fc2_1 = nn.Linear(256, 1)
        self.fc2_2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        cond = torch.sigmoid(self.fc2_1(x))
        offset = self.fc2_2(x)
        return cond, offset


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms

    path = r"E:\celeba\smallset\12\positive/0.jpg"
    img = Image.open(path)
    transform = transforms.ToTensor()
    input = transform(img)
    # print(input.shape)
    x = input.unsqueeze(0)
    # print(x.shape)

    net = pnet()
    cond, offset = net(x)
    print(cond)
    print("////////////////////////////")
    print(offset)