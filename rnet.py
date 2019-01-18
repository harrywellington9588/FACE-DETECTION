import torch.nn as nn
import torch


class rnet(nn.Module):
    def __init__(self):
        super(rnet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(28, 48, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*2*2, 128),
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