import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetBN(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(SimpleNetBN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=2, bias=False
        )
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=2, bias=False
        )
        self.bn_conv2 = nn.BatchNorm2d(16)

        self.linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=120, bias=False)
        self.bn_linear1 = nn.BatchNorm1d(120)
        self.linear2 = nn.Linear(in_features=120, out_features=84, bias=False)
        self.bn_linear2 = nn.BatchNorm1d(84)
        self.linear3 = nn.Linear(in_features=84, out_features=num_classes, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)
        x = self.bn_linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn_linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
