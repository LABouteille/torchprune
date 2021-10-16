import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc


class TestDependencyGraph:
    @classmethod
    def setup_class(cls):
        cls._seed_everything(42)

        cls.model_sequential = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1),
        )

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
                self.conv2 = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                return x

        cls.model_class = ConvNet()

    @classmethod
    def teardown_class(cls):
        del cls.model_sequential
        del cls.model_class

    @classmethod
    def _seed_everything(cls, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def test_build_dependency_graph_sequential(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph()
        DG.build_dependency_graph(self.model_sequential, x)

        print(DG.graph_)

    def test_build_dependency_graph_class(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph()
        DG.build_dependency_graph(self.model_class, x)

        print(DG.graph_)
