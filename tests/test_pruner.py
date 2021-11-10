import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc

# from unittest.mock import Mock

# from torchcompress.node import OPTYPE, Node


class TestPruner:
    @classmethod
    def setup_class(cls):
        cls._seed_everything(42)

        class NeuralNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
                self.conv2 = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
                self.conv3 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=1)

                self.linear1 = nn.Linear(in_features=16, out_features=8)
                self.linear2 = nn.Linear(in_features=8, out_features=4)
                self.linear3 = nn.Linear(in_features=4, out_features=3)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = self.conv3(x)
                x = self.act3(x)

                x = torch.flatten(x)

                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = F.relu(x)
                x = self.linear3(x)
                x = F.relu(x)
                return x

        cls.model = NeuralNet()
        x = torch.randn(1, 2, 4, 4)
        cls.DG = tc.DependencyGraph(cls.model)
        cls.DG.build_dependency_graph(x)

    @classmethod
    def teardown_class(cls):
        del cls.model
        del cls.DG

    @classmethod
    def _seed_everything(cls, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
