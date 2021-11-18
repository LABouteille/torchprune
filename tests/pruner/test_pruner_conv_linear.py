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

    @pytest.fixture(scope="function")
    def setup_and_teardown_no_stride(self):

        self._seed_everything(42)

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1)
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
                self.linear1 = nn.Linear(in_features=16, out_features=8)

            def forward(self, x):
                x = self.conv0(x)
                x = F.relu(x)
                x = self.conv1(x)
                x = F.relu(x)
                x = torch.flatten(x, start_dim=1)
                x = self.linear1(x)
                x = F.relu(x)
                return x

        # Setup
        self.model = SimpleNet()
        self.x = torch.randn(1, 2, 4, 4)
        self.DG = tc.DependencyGraph(self.model)
        self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teadown
        del self.model
        del self.DG

    @pytest.fixture(scope="function")
    def setup_and_teardown_stride(self):

        self._seed_everything(42)

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    in_channels=6, out_channels=16, kernel_size=5, stride=2
                )
                self.linear1 = nn.Linear(in_features=16 * 5 * 5, out_features=2)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = torch.flatten(x, start_dim=1)
                x = self.linear1(x)
                x = F.relu(x)
                return x

        # Setup
        self.model = SimpleNet()
        self.x = torch.randn(1, 6, 14, 14)
        self.DG = tc.DependencyGraph(self.model)
        self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teadown
        del self.model
        del self.DG

    @pytest.mark.usefixtures("setup_and_teardown_no_stride")
    def test_expand_indices_1(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)
        node = self.DG.module_to_node[self.model.conv1]
        new_indices = pruner._Pruner__expand_indices(input_node=node, indices=[0])
        assert np.array_equal(new_indices, [0, 1, 2, 3])

    @pytest.mark.usefixtures("setup_and_teardown_no_stride")
    def test_expand_indices_2(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)
        node = self.DG.module_to_node[self.model.conv1]
        new_indices = pruner._Pruner__expand_indices(input_node=node, indices=[0, 3])
        assert np.array_equal(new_indices, [0, 1, 2, 3, 12, 13, 14, 15])

    @pytest.mark.usefixtures("setup_and_teardown_no_stride")
    def test_flatten_1(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.conv1.in_channels == 3
        assert self.model.conv1.out_channels == 4
        assert self.model.conv1.bias.shape == torch.Size([4])
        assert self.model.linear1.in_features == 16
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv1.in_channels == 3
        assert self.model.conv1.out_channels == 3
        assert self.model.conv1.bias.shape == torch.Size([3])
        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])

        assert self.model(self.x).shape == torch.Size([1, 8])

    @pytest.mark.usefixtures("setup_and_teardown_no_stride")
    def test_flatten_2(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.conv1.in_channels == 3
        assert self.model.conv1.out_channels == 4
        assert self.model.conv1.bias.shape == torch.Size([4])
        assert self.model.linear1.in_features == 16
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.75
        )

        assert self.model.conv1.in_channels == 3
        assert self.model.conv1.out_channels == 1
        assert self.model.conv1.bias.shape == torch.Size([1])
        assert self.model.linear1.in_features == 4
        assert self.model.linear1.out_features == 8
        assert self.model.linear1.bias.shape == torch.Size([8])

        assert self.model(self.x).shape == torch.Size([1, 8])

    @pytest.mark.usefixtures("setup_and_teardown_stride")
    def test(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.conv1.in_channels == 6
        assert self.model.conv1.out_channels == 16
        assert self.model.conv1.bias.shape == torch.Size([16])
        assert self.model.linear1.in_features == 16 * 5 * 5
        assert self.model.linear1.out_features == 2
        assert self.model.linear1.bias.shape == torch.Size([2])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.7
        )

        assert self.model.conv1.in_channels == 6
        assert self.model.conv1.out_channels == 5
        assert self.model.conv1.bias.shape == torch.Size([5])
        assert self.model.linear1.in_features == 5 * 5 * 5
        assert self.model.linear1.out_features == 2
        assert self.model.linear1.bias.shape == torch.Size([2])

        assert self.model(self.x).shape == torch.Size([1, 2])
