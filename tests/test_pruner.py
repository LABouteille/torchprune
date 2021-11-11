import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc


class TestPruner:
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
                x = F.relu(x)

                x = torch.flatten(x)

                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = F.relu(x)
                x = self.linear3(x)
                x = F.relu(x)
                return x

        # Setup
        self.model = NeuralNet()
        self.x = torch.randn(1, 2, 4, 4)
        self.DG = tc.DependencyGraph(self.model)
        self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teadown
        del self.model
        del self.DG

    def test_pruner(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        # Prune conv1
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

        assert self.model.conv1(self.x).shape == torch.Size([1, 3, 2, 2])
        prev_res = self.model.conv1(self.x)

        # Prune conv2
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

        assert self.model.conv2(prev_res).shape == torch.Size([1, 4, 2, 2])
        prev_res = self.model.conv2(prev_res)

        # Prune conv3
        assert self.model.conv3.in_channels == 4
        assert self.model.conv3.out_channels == 4
        assert self.model.conv3.bias.shape == torch.Size([4])
        assert self.model.linear1.in_features == 16
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])

        pruner.run(
            layer=self.model.conv3, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv3.in_channels == 4
        assert self.model.conv3.out_channels == 3
        assert self.model.conv3.bias.shape == torch.Size([3])
        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])

        assert self.model.conv3(prev_res).shape == torch.Size([1, 3, 2, 2])
        prev_res = self.model.conv3(prev_res)

        # Prune linear1
        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])
        assert self.model.linear2.in_features == 8
        assert self.model.linear2.out_features == 4
        assert self.model.linear2.bias.shape == torch.Size([4])

        pruner.run(
            layer=self.model.linear1, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 7
        assert self.model.linear1.bias.shape == torch.Size([7])
        assert self.model.linear2.in_features == 7
        assert self.model.linear2.out_features == 4
        assert self.model.linear2.bias.shape == torch.Size([4])

        assert self.model.linear1(torch.flatten(prev_res)).shape == torch.Size([7])
        prev_res = self.model.linear1(torch.flatten(prev_res))

        # Prune linear2
        assert self.model.linear2.in_features == 7
        assert self.model.linear2.out_features == 4
        assert self.model.linear2.bias.shape == torch.Size([4])
        assert self.model.linear3.in_features == 4
        assert self.model.linear3.out_features == 3
        assert self.model.linear3.bias.shape == torch.Size([3])

        pruner.run(
            layer=self.model.linear2, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.linear2.in_features == 7
        assert self.model.linear2.out_features == 3
        assert self.model.linear2.bias.shape == torch.Size([3])
        assert self.model.linear3.in_features == 3
        assert self.model.linear3.out_features == 3
        assert self.model.linear3.bias.shape == torch.Size([3])

        assert self.model.linear2(prev_res).shape == torch.Size([3])
        prev_res = self.model.linear2(prev_res)

        # Prune linear3
        assert self.model.linear3.in_features == 3
        assert self.model.linear3.out_features == 3
        assert self.model.linear3.bias.shape == torch.Size([3])

        pruner.run(
            layer=self.model.linear3, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.linear3.in_features == 3
        assert self.model.linear3.out_features == 2
        assert self.model.linear3.bias.shape == torch.Size([2])

        assert self.model.linear3(prev_res).shape == torch.Size([2])

        assert self.model(self.x).shape == torch.Size([2])
