import os
import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc
from torchcompress.node import OPTYPE, Node


class TestDependencyGraph:
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

        yield  # Test will be run here

        # Teardown
        del self.model

    def test_build_graph(self):
        x = torch.randn(1, 2, 4, 4)
        DG = tc.DependencyGraph(self.model)
        module_to_node = DG._DependencyGraph__build_graph(x)

        module_to_node_mock = Mock()
        module_to_node_copy = module_to_node.copy()

        def getitem(module):
            return module_to_node_copy[module]

        def setitem(module, node):
            module_to_node_copy[module] = node

        module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
        module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model.modules())
        conv1 = list_modules[1]
        conv2 = list_modules[2]
        conv3 = list_modules[3]
        linear1 = list_modules[4]
        linear2 = list_modules[5]
        linear3 = list_modules[6]
        relu = None

        module_to_node_mock[conv1] = Node(
            module=conv1, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[relu] = Node(
            module=relu, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        module_to_node_mock[conv2] = Node(
            module=conv2, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv3] = Node(
            module=conv3, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[linear1] = Node(
            module=linear1, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear2] = Node(
            module=linear2, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear3] = Node(
            module=linear3, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        for module, node in module_to_node.items():
            node_mock = module_to_node_mock[module]
            assert node.module == node_mock.module

    def test_order_dependency_graph(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph(self.model)
        module_to_node = DG._DependencyGraph__build_graph(x)
        ordered_node = DG._DependencyGraph__order_dependency_graph(module_to_node)

        module_to_node_mock = Mock()
        module_to_node_copy = module_to_node.copy()

        def getitem(module):
            return module_to_node_copy[module]

        def setitem(module, node):
            module_to_node_copy[module] = node

        module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
        module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model.modules())
        conv1 = list_modules[1]
        conv2 = list_modules[2]
        conv3 = list_modules[3]
        linear1 = list_modules[4]
        linear2 = list_modules[5]
        linear3 = list_modules[6]
        relu = None
        flatten = None

        module_to_node_mock[conv1] = Node(
            module=conv1, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[conv2] = Node(
            module=conv2, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv3] = Node(
            module=conv3, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[linear1] = Node(
            module=linear1, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear2] = Node(
            module=linear2, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear3] = Node(
            module=linear3, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[relu] = Node(
            module=relu, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )

        module_to_node_mock[flatten] = Node(
            module=flatten, op_type=OPTYPE.FLATTEN, grad_fn=lambda: None
        )

        ordered_node_mock = [
            module_to_node_mock[conv1],
            module_to_node_mock[relu],
            module_to_node_mock[conv2],
            module_to_node_mock[relu],
            module_to_node_mock[conv3],
            module_to_node_mock[relu],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[linear1],
            module_to_node_mock[relu],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[linear2],
            module_to_node_mock[relu],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[flatten],
            module_to_node_mock[linear3],
            module_to_node_mock[relu],
        ]

        for i, node in enumerate(ordered_node):
            node_mock = ordered_node_mock[i]
            assert node.module == node_mock.module

    def test_build_dependency(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph(self.model)
        module_to_node = DG._DependencyGraph__build_graph(x)
        ordered_node = DG._DependencyGraph__order_dependency_graph(module_to_node)
        dependencies = DG._DependencyGraph__build_dependency(ordered_node)

        # Mock graph
        module_to_node_mock = Mock()
        module_to_node_copy = module_to_node.copy()

        def getitem(module):
            return module_to_node_copy[module]

        def setitem(module, node):
            module_to_node_copy[module] = node

        module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
        module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model.modules())
        conv1 = list_modules[1]
        conv2 = list_modules[2]
        conv3 = list_modules[3]
        linear1 = list_modules[4]
        linear2 = list_modules[5]
        linear3 = list_modules[6]
        relu = None
        flatten = None

        module_to_node_mock[conv1] = Node(
            module=conv1, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[conv2] = Node(
            module=conv2, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv3] = Node(
            module=conv3, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[linear1] = Node(
            module=linear1, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear2] = Node(
            module=linear2, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear3] = Node(
            module=linear3, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[relu] = Node(
            module=relu, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )

        module_to_node_mock[flatten] = Node(
            module=flatten, op_type=OPTYPE.FLATTEN, grad_fn=lambda: None
        )

        dependencies_mock = {
            module_to_node_mock[conv1]: [
                module_to_node_mock[relu],
                module_to_node_mock[conv2],
            ],
            module_to_node_mock[conv2]: [
                module_to_node_mock[relu],
                module_to_node_mock[conv3],
            ],
            module_to_node_mock[conv3]: [
                module_to_node_mock[relu],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[linear1],
            ],
            module_to_node_mock[linear1]: [
                module_to_node_mock[relu],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[linear2],
            ],
            module_to_node_mock[linear2]: [
                module_to_node_mock[relu],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[linear3],
            ],
            module_to_node_mock[linear3]: [module_to_node_mock[relu]],
        }

        for node, dep in dependencies.items():
            node_mock = module_to_node_mock[node.module]
            assert node.module == node_mock.module
            assert len(dep) == len(dependencies_mock[node_mock])
            for dep_node, dep_node_mock in zip(dep, dependencies_mock[node_mock]):
                assert dep_node.module == dep_node_mock.module

    def test_clean_dependency(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph(self.model)
        module_to_node = DG._DependencyGraph__build_graph(x)
        ordered_node = DG._DependencyGraph__order_dependency_graph(module_to_node)
        dependencies = DG._DependencyGraph__build_dependency(ordered_node)
        DG._DependencyGraph__clean_dependency(dependencies)

        # Mock graph
        module_to_node_mock = Mock()
        module_to_node_copy = module_to_node.copy()

        def getitem(module):
            return module_to_node_copy[module]

        def setitem(module, node):
            module_to_node_copy[module] = node

        module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
        module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model.modules())
        conv1 = list_modules[1]
        conv2 = list_modules[2]
        conv3 = list_modules[3]
        linear1 = list_modules[4]
        linear2 = list_modules[5]
        linear3 = list_modules[6]
        relu = None
        flatten = None

        module_to_node_mock[conv1] = Node(
            module=conv1, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[conv2] = Node(
            module=conv2, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv3] = Node(
            module=conv3, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        module_to_node_mock[linear1] = Node(
            module=linear1, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear2] = Node(
            module=linear2, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[linear3] = Node(
            module=linear3, op_type=OPTYPE.LINEAR, grad_fn=lambda: None
        )

        module_to_node_mock[relu] = Node(
            module=relu, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )

        module_to_node_mock[flatten] = Node(
            module=flatten, op_type=OPTYPE.FLATTEN, grad_fn=lambda: None
        )

        dependencies_mock = {
            module_to_node_mock[conv1]: [
                module_to_node_mock[relu],
                module_to_node_mock[conv2],
            ],
            module_to_node_mock[conv2]: [
                module_to_node_mock[relu],
                module_to_node_mock[conv3],
            ],
            module_to_node_mock[conv3]: [
                module_to_node_mock[relu],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[flatten],
                module_to_node_mock[linear1],
            ],
            module_to_node_mock[linear1]: [
                module_to_node_mock[relu],
                module_to_node_mock[linear2],
            ],
            module_to_node_mock[linear2]: [
                module_to_node_mock[relu],
                module_to_node_mock[linear3],
            ],
            module_to_node_mock[linear3]: [module_to_node_mock[relu]],
        }

        for node, dep in dependencies.items():
            node_mock = module_to_node_mock[node.module]
            assert node.module == node_mock.module
            assert len(dep) == len(dependencies_mock[node_mock])
            for dep_node, dep_node_mock in zip(dep, dependencies_mock[node_mock]):
                assert dep_node.module == dep_node_mock.module
