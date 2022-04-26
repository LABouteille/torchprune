import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchcompress as tc


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SkipConnect(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(SkipConnect, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != expansion * out_channels:
            print("HELLO")
            self.shortcut = nn.Conv2d(
                in_channels, expansion * out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += self.shortcut(identity)
        out = F.relu(out)
        return out


class TestPrunerConv:
    @classmethod
    def _seed_everything(cls, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    @pytest.fixture(scope="function")
    def setup_and_teardown_test1(self):
        self._seed_everything(42)

        # Setup
        self.model = SkipConnect(16, 16)
        self.x = torch.randn(1, 16, 32, 32)
        # self.DG = tc.DependencyGraph(self.model)
        # self.DG.build_dependency_graph(self.x)

        # from torchviz import make_dot

        # g = make_dot(self.model(self.x), params=dict(self.model.named_parameters()))
        # g.render(filename="graph")

        yield  # Test will be run here

        # Teardown
        del self.model
        # del self.DG

    @pytest.fixture(scope="function")
    def setup_and_teardown_test2(self):
        self._seed_everything(42)

        # Setup
        self.model = SkipConnect(16, 16)
        self.x = torch.randn(1, 16, 32, 32)
        # self.DG = tc.DependencyGraph(self.model)
        # self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teardown
        del self.model
        # del self.DG

    @pytest.mark.usefixtures("setup_and_teardown_test1")
    def test1(self):
        print(self.model(self.x).shape)

    @pytest.mark.usefixtures("setup_and_teardown_test2")
    def test2(self):
        pass
