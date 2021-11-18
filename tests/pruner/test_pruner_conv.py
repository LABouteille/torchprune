import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc


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

    @pytest.fixture(autouse=True, scope="function")
    def setup_and_teardown_at_each_test(self):

        self._seed_everything(42)

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

        # Setup
        self.model = ConvNet()
        self.x = torch.randn(1, 2, 4, 4)
        self.DG = tc.DependencyGraph(self.model)
        self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teardown
        del self.model
        del self.DG

    def test_conv_1(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.conv1.in_channels == 2
        assert self.model.conv1.out_channels == 4
        assert self.model.conv1.bias.shape == torch.Size([4])
        assert self.model.conv2.in_channels == 4
        assert self.model.conv2.out_channels == 5
        assert self.model.conv2.bias.shape == torch.Size([5])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv1.in_channels == 2
        assert self.model.conv1.out_channels == 3
        assert self.model.conv1.bias.shape == torch.Size([3])
        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 5
        assert self.model.conv2.bias.shape == torch.Size([5])

        assert self.model.conv2(self.model.conv1(self.x)).shape == torch.Size(
            [1, 5, 2, 2]
        )

    def test_conv_2(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.conv1.in_channels == 2
        assert self.model.conv1.out_channels == 4
        assert self.model.conv1.bias.shape == torch.Size([4])
        assert self.model.conv2.in_channels == 4
        assert self.model.conv2.out_channels == 5
        assert self.model.conv2.bias.shape == torch.Size([5])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv1.in_channels == 2
        assert self.model.conv1.out_channels == 3
        assert self.model.conv1.bias.shape == torch.Size([3])
        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 5
        assert self.model.conv2.bias.shape == torch.Size([5])

        assert self.model.conv2(self.model.conv1(self.x)).shape == torch.Size(
            [1, 5, 2, 2]
        )

        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 5
        assert self.model.conv2.bias.shape == torch.Size([5])
        assert self.model.conv3.in_channels == 5
        assert self.model.conv3.out_channels == 4
        assert self.model.conv3.bias.shape == torch.Size([4])

        pruner.run(
            layer=self.model.conv2, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv2.in_channels == 3
        assert self.model.conv2.out_channels == 4
        assert self.model.conv2.bias.shape == torch.Size([4])
        assert self.model.conv3.in_channels == 4
        assert self.model.conv3.out_channels == 4
        assert self.model.conv3.bias.shape == torch.Size([4])

        assert self.model(self.x).shape == torch.Size([1, 4, 2, 2])
