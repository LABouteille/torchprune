import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc


class TestPrunerBatchnorm:
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
    def setup_and_teardown_for_batchnorm2d(self):

        self._seed_everything(42)

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(
                    in_channels=2, out_channels=4, kernel_size=3, bias=False
                )
                self.bn1 = nn.BatchNorm2d(4)
                self.conv2 = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
                self.bn2 = nn.BatchNorm2d(5, affine=False)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = F.relu(x)
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

    @pytest.fixture(scope="function")
    def setup_and_teardown_for_batchnorm1d(self):
        self._seed_everything(42)

        class LinearNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(in_features=10, out_features=8, bias=False)
                self.bn1 = nn.BatchNorm1d(8)
                self.linear2 = nn.Linear(in_features=8, out_features=4)
                self.bn2 = nn.BatchNorm1d(4, affine=False)

            def forward(self, x):
                x = self.linear1(x)
                x = self.bn1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = self.bn2(x)
                x = F.relu(x)
                return x

        # Setup
        self.model = LinearNet()
        self.x = torch.randn(2, 10)
        self.DG = tc.DependencyGraph(self.model)
        self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teardown
        del self.model
        del self.DG

    @pytest.fixture(scope="function")
    def setup_and_teardown_for_batchnorm(self):
        self._seed_everything(42)

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=1)
                self.conv1 = nn.Conv2d(
                    in_channels=3, out_channels=4, kernel_size=3, bias=False
                )
                self.bn_conv1 = nn.BatchNorm2d(4)
                self.linear1 = nn.Linear(in_features=16, out_features=8, bias=False)
                self.bn_linear1 = nn.BatchNorm1d(8)

            def forward(self, x):
                x = self.conv0(x)
                x = F.relu(x)
                x = self.conv1(x)
                x = self.bn_conv1(x)
                x = F.relu(x)
                x = torch.flatten(x, start_dim=1)
                x = self.linear1(x)
                x = self.bn_linear1(x)
                x = F.relu(x)
                return x

        # Setup
        self.model = SimpleNet()
        self.x = torch.randn(
            2, 2, 4, 4
        )  # Batchnorm1d requires > 1 sample to compute running_mean/running_variance
        self.DG = tc.DependencyGraph(self.model)
        self.DG.build_dependency_graph(self.x)

        yield  # Test will be run here

        # Teardown
        del self.model
        del self.DG

    @pytest.mark.usefixtures("setup_and_teardown_for_batchnorm2d")
    def test_batchnorm2d_affine(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.bn1.num_features == 4
        assert self.model.bn1.running_mean.shape == torch.Size([4])
        assert self.model.bn1.running_var.shape == torch.Size([4])
        assert self.model.bn1.weight.shape == torch.Size([4])
        assert self.model.bn1.bias.shape == torch.Size([4])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.bn1.num_features == 3
        assert self.model.bn1.running_mean.shape == torch.Size([3])
        assert self.model.bn1.running_var.shape == torch.Size([3])
        assert self.model.bn1.weight.shape == torch.Size([3])
        assert self.model.bn1.bias.shape == torch.Size([3])

        assert self.model.bn1(self.model.conv1(self.x)).shape == torch.Size(
            [1, 3, 2, 2]
        )

    @pytest.mark.usefixtures("setup_and_teardown_for_batchnorm2d")
    def test_batchnorm2d_not_affine(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.bn1.num_features == 4
        assert self.model.bn1.running_mean.shape == torch.Size([4])
        assert self.model.bn1.running_var.shape == torch.Size([4])
        assert self.model.bn1.weight.shape == torch.Size([4])
        assert self.model.bn1.bias.shape == torch.Size([4])

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.bn1.num_features == 3
        assert self.model.bn1.running_mean.shape == torch.Size([3])
        assert self.model.bn1.running_var.shape == torch.Size([3])
        assert self.model.bn1.weight.shape == torch.Size([3])
        assert self.model.bn1.bias.shape == torch.Size([3])

        assert self.model.bn1(self.model.conv1(self.x)).shape == torch.Size(
            [1, 3, 2, 2]
        )
        prev_res = self.model.bn1(self.model.conv1(self.x))

        # Batchnorm affine = False
        assert self.model.bn2.num_features == 5
        assert self.model.bn2.running_mean.shape == torch.Size([5])
        assert self.model.bn2.running_var.shape == torch.Size([5])

        pruner.run(
            layer=self.model.conv2, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.bn2.num_features == 4
        assert self.model.bn2.running_mean.shape == torch.Size([4])
        assert self.model.bn2.running_var.shape == torch.Size([4])

        assert self.model.bn2(self.model.conv2(prev_res)).shape == torch.Size(
            [1, 4, 2, 2]
        )

    @pytest.mark.usefixtures("setup_and_teardown_for_batchnorm1d")
    def test_batchnorm1d_affine(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.bn1.num_features == 8
        assert self.model.bn1.running_mean.shape == torch.Size([8])
        assert self.model.bn1.running_var.shape == torch.Size([8])
        assert self.model.bn1.weight.shape == torch.Size([8])
        assert self.model.bn1.bias.shape == torch.Size([8])

        pruner.run(
            layer=self.model.linear1, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.bn1.num_features == 7
        assert self.model.bn1.running_mean.shape == torch.Size([7])
        assert self.model.bn1.running_var.shape == torch.Size([7])
        assert self.model.bn1.weight.shape == torch.Size([7])
        assert self.model.bn1.bias.shape == torch.Size([7])

        assert self.model.bn1(self.model.linear1(self.x)).shape == torch.Size([2, 7])

    @pytest.mark.usefixtures("setup_and_teardown_for_batchnorm1d")
    def test_batchnorm1d_not_affine(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.bn1.num_features == 8
        assert self.model.bn1.running_mean.shape == torch.Size([8])
        assert self.model.bn1.running_var.shape == torch.Size([8])
        assert self.model.bn1.weight.shape == torch.Size([8])
        assert self.model.bn1.bias.shape == torch.Size([8])

        pruner.run(
            layer=self.model.linear1, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.bn1.num_features == 7
        assert self.model.bn1.running_mean.shape == torch.Size([7])
        assert self.model.bn1.running_var.shape == torch.Size([7])
        assert self.model.bn1.weight.shape == torch.Size([7])
        assert self.model.bn1.bias.shape == torch.Size([7])

        assert self.model.bn1(self.model.linear1(self.x)).shape == torch.Size([2, 7])
        prev_res = self.model.bn1(self.model.linear1(self.x))

        # Batchnorm affine = False
        assert self.model.bn2.num_features == 4
        assert self.model.bn2.running_mean.shape == torch.Size([4])
        assert self.model.bn2.running_var.shape == torch.Size([4])

        pruner.run(
            layer=self.model.linear2, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.bn2.num_features == 3
        assert self.model.bn2.running_mean.shape == torch.Size([3])
        assert self.model.bn2.running_var.shape == torch.Size([3])

        assert self.model.bn2(self.model.linear2(prev_res)).shape == torch.Size([2, 3])

    @pytest.mark.usefixtures("setup_and_teardown_for_batchnorm")
    def test_batchnorm(self):
        pruner = tc.Pruner(DG=self.DG, dummy_input=self.x)

        assert self.model.conv1.in_channels == 3
        assert self.model.conv1.out_channels == 4
        assert self.model.bn_conv1.num_features == 4
        assert self.model.bn_conv1.running_mean.shape == torch.Size([4])
        assert self.model.bn_conv1.running_var.shape == torch.Size([4])
        assert self.model.bn_conv1.weight.shape == torch.Size([4])
        assert self.model.bn_conv1.bias.shape == torch.Size([4])
        assert self.model.linear1.in_features == 16
        assert self.model.linear1.out_features == 8

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv1.in_channels == 3
        assert self.model.conv1.out_channels == 3
        assert self.model.bn_conv1.num_features == 3
        assert self.model.bn_conv1.running_mean.shape == torch.Size([3])
        assert self.model.bn_conv1.running_var.shape == torch.Size([3])
        assert self.model.bn_conv1.weight.shape == torch.Size([3])
        assert self.model.bn_conv1.bias.shape == torch.Size([3])
        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 8

        assert self.model.bn_conv1(
            self.model.conv1(self.model.conv0(self.x))
        ).shape == torch.Size([2, 3, 2, 2])

        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 8
        assert self.model.bn_linear1.num_features == 8
        assert self.model.bn_linear1.running_mean.shape == torch.Size([8])
        assert self.model.bn_linear1.running_var.shape == torch.Size([8])
        assert self.model.bn_linear1.weight.shape == torch.Size([8])
        assert self.model.bn_linear1.bias.shape == torch.Size([8])

        pruner.run(
            layer=self.model.linear1, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 7
        assert self.model.bn_linear1.num_features == 7
        assert self.model.bn_linear1.running_mean.shape == torch.Size([7])
        assert self.model.bn_linear1.running_var.shape == torch.Size([7])
        assert self.model.bn_linear1.weight.shape == torch.Size([7])
        assert self.model.bn_linear1.bias.shape == torch.Size([7])

        assert self.model(self.x).shape == torch.Size([2, 7])
