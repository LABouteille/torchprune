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

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
                self.conv2 = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
                self.conv3 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=1)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = self.conv3(x)
                return x

        cls.model = ConvNet()
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

    def test_pruner(self):
        pruner = tc.Pruner(self.DG)

        assert self.model.conv1.in_channels == 2
        assert self.model.conv1.out_channels == 4
        assert self.model.conv2.in_channels == 4
        assert self.model.conv2.out_channels == 5

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_strategy, amount_to_prune=0.25
        )

        assert self.model.conv1.in_channels == 2
        assert self.model.conv1.out_channels == 3
        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 5

        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 5
        assert self.model.conv3.in_channels == 5
        assert self.model.conv3.out_channels == 4

        pruner.run(
            layer=self.model.conv2, criteria=tc.random_strategy, amount_to_prune=0.25
        )

        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 4
        assert self.model.conv3.in_channels == 4
        assert self.model.conv3.out_channels == 4
