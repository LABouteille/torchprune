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
                x = torch.flatten(x)
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

    def test_expand_indices_1(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)
        node = self.DG.module_to_node[self.model.conv1]
        nb_filter_before_pruning = self.model.conv1.weight.shape[0]
        new_indices = pruner._Pruner__expand_indices(
            input_node=node,
            indices=[0],
            nb_filter_before_pruning=nb_filter_before_pruning,
        )
        assert np.array_equal(new_indices, [0, 1, 2, 3])

    def test_expand_indices_2(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)
        node = self.DG.module_to_node[self.model.conv1]
        nb_filter_before_pruning = self.model.conv1.weight.shape[0]
        new_indices = pruner._Pruner__expand_indices(
            input_node=node,
            indices=[0, 3],
            nb_filter_before_pruning=nb_filter_before_pruning,
        )
        assert np.array_equal(new_indices, [0, 1, 2, 3, 12, 13, 14, 15])

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

        assert self.model(self.x).shape == torch.Size([8])

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

        assert self.model(self.x).shape == torch.Size([8])
