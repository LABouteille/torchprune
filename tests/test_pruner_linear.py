import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc

# from unittest.mock import Mock

# from torchcompress.node import OPTYPE, Node


class TestPrunerLinear:
    @classmethod
    def setup_class(cls):
        cls._seed_everything(42)

        class LinearNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(in_features=10, out_features=8)
                self.linear2 = nn.Linear(in_features=8, out_features=4)
                self.linear3 = nn.Linear(in_features=4, out_features=3)

            def forward(self, x):
                x = self.linear1(x)
                x = F.relu(x)
                x = self.linear2(x)
                x = F.relu(x)
                x = self.linear3(x)
                return x

        cls.model = LinearNet()
        x = torch.randn(2, 10)
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

    def test_linear(self):

        pruner = tc.Pruner(self.DG)

        assert self.model.linear1.in_features == 10
        assert self.model.linear1.out_features == 8
        assert self.model.linear2.in_features == 8
        assert self.model.linear2.out_features == 4

        pruner.run(
            layer=self.model.linear1, criteria=tc.random_criteria, amount_to_prune=0.1
        )

        assert self.model.linear1.in_features == 10
        assert self.model.linear1.out_features == 7
        assert self.model.linear2.in_features == 7
        assert self.model.linear2.out_features == 4

        assert self.model.linear2.in_features == 7
        assert self.model.linear2.out_features == 4
        assert self.model.linear3.in_features == 4
        assert self.model.linear3.out_features == 3

        pruner.run(
            layer=self.model.linear2, criteria=tc.random_criteria, amount_to_prune=0.5
        )

        assert self.model.linear2.in_features == 7
        assert self.model.linear2.out_features == 2
        assert self.model.linear3.in_features == 2
        assert self.model.linear3.out_features == 3
