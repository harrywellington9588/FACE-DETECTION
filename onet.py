import torch.nn as nn
import torch


class onet(nn.Module):
    def __init__(self):
        super(onet, self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 256, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256*2*2, 256),
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