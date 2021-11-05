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

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Never prune first convolution. Otherwise we have to reshape input data.
                self.conv0 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1)
                self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3)
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

        class NeuralNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
                self.act1 = nn.ReLU()
                self.conv2 = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
                self.act2 = nn.ReLU()
                self.conv3 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=1)
                self.act3 = nn.ReLU()

                self.linear1 = nn.Linear(in_features=16, out_features=8)
                self.act4 = nn.ReLU()
                self.linear2 = nn.Linear(in_features=8, out_features=4)
                self.act5 = nn.ReLU()
                self.linear3 = nn.Linear(in_features=4, out_features=3)
                self.act6 = nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.act1(x)
                # x = F.relu(x)
                x = self.conv2(x)
                x = self.act2(x)
                # x = F.relu(x)
                x = self.conv3(x)
                x = self.act3(x)

                x = torch.flatten(x)

                x = self.linear1(x)
                x = self.act4(x)
                # x = F.relu(x)
                x = self.linear2(x)
                x = self.act5(x)
                # x = F.relu(x)
                x = self.linear3(x)
                x = self.act6(x)
                return x

        from torchviz import make_dot

        cls.model = SimpleNet()
        x = torch.randn(1, 2, 4, 4)
        g = make_dot(cls.model(x))
        g.render(filename="graph-full")

        cls.DG = tc.DependencyGraph(cls.model)
        cls.DG.build_dependency_graph(x)
        # print()

        # print("===GRAPH===")
        # for key, value in cls.DG.module_to_node.items():
        #     print(key)
        #     print(f"\t{value.outputs}")
        #     print('----')

        print("===DEPENDENCIES===")
        for key, value in cls.DG.dependencies.items():
            print(key.module)
            for val in value:
                print(f"\t{val}")
            print("----")

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

    def test_flatten(self):
        x = torch.randn(1, 2, 4, 4)
        pruner = tc.Pruner(DG=self.DG, dummy_input=x)

        assert self.model.conv1.in_channels == 4
        assert self.model.conv1.out_channels == 4
        assert self.model.linear1.in_features == 16
        assert self.model.linear1.out_features == 8

        pruner.run(
            layer=self.model.conv1, criteria=tc.random_criteria, amount_to_prune=0.25
        )

        assert self.model.conv1.in_channels == 4
        assert self.model.conv1.out_channels == 3
        assert self.model.linear1.in_features == 12
        assert self.model.linear1.out_features == 8
