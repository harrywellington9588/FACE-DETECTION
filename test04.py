from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class pnet(nn.Module):
    def __init__(self):
        super(pnet,self).__init__()
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
        cond = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cond, offset


class rnet(nn.Module):
    def __init__(self):
        super(rnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.PReLU()
        )
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(28, 48, 3),
            nn.PReLU()
        )
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, 2),
            nn.PReLU()
        )
        # self.pre_layer = nn.Sequential(
        #     nn.Conv2d(3, 28, 3),
        #     nn.PReLU(),
        #     nn.MaxPool2d(3, 2),
        #     nn.Conv2d(28, 48, 3),
        #     nn.PReLU(),
        #     nn.MaxPool2d(3, 2),
        #     nn.Conv2d(48, 64, 2),
        #     nn.PReLU()
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(64*2*2, 128),
            nn.PReLU()
        )
        self.fc2_1 = nn.Linear(128, 1)
        self.fc2_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        print(x.size())
        print("1")
        x = self.pool1(x)
        print(x.size())
        print("2")
        x = self.conv2(x)
        print(x.size())
        print("3")
        x = self.pool2(x)
        print(x.size())
        print("4")
        x = self.conv3(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.fc1(x)
        print(x.size())
        cond = F.sigmoid(self.fc2_1(x))
        offset = self.fc2_2(x)
        return cond, offset


if __name__ == '__main__':
    net = rnet()
    x = np.arange(1 * 3 * 24 * 24).reshape(1, 3, 24, 24)
    x = torch.Tensor(x)
    cond, offset = net(x)
    print(cond.shape, offset.shape)