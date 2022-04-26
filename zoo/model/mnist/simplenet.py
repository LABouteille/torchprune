import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=4, stride=2)

        self.linear1 = nn.Linear(in_features=5 * 2 * 2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
