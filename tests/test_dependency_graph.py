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
        conv_2_4_module = list_modules[1]
        relu_module = None
        conv_4_5_module = list_modules[2]
        conv_5_4_module = list_modules[3]

        module_to_node_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        module_to_node_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv_5_4_module] = Node(
            module=conv_5_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
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
        conv_2_4_module = list_modules[1]
        relu_module = None
        conv_4_5_module = list_modules[2]
        conv_5_4_module = list_modules[3]

        module_to_node_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        module_to_node_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv_5_4_module] = Node(
            module=conv_5_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        ordered_node_mock = [
            module_to_node_mock[conv_2_4_module],
            module_to_node_mock[relu_module],
            module_to_node_mock[conv_4_5_module],
            module_to_node_mock[relu_module],
            module_to_node_mock[conv_5_4_module],
        ]

        for i, node in enumerate(ordered_node):
            node_mock = ordered_node_mock[i]
            assert node.module == node_mock.module

    def test_build_dependency_graph(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph(self.model)
        DG.build_dependency_graph(x)

        # Mock graph
        module_to_node_mock = Mock()
        module_to_node_copy = DG.module_to_node.copy()

        def getitem(module):
            return module_to_node_copy[module]

        def setitem(module, node):
            module_to_node_copy[module] = node

        module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
        module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model.modules())
        conv_2_4_module = list_modules[1]
        relu_module = None
        conv_4_5_module = list_modules[2]
        conv_5_4_module = list_modules[3]

        module_to_node_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        module_to_node_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        module_to_node_mock[conv_5_4_module] = Node(
            module=conv_5_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        dependencies_mock = {
            module_to_node_mock[conv_2_4_module]: [
                module_to_node_mock[relu_module],
                module_to_node_mock[conv_4_5_module],
            ],
            module_to_node_mock[conv_4_5_module]: [
                module_to_node_mock[relu_module],
                module_to_node_mock[conv_5_4_module],
            ],
            module_to_node_mock[conv_5_4_module]: [],
        }

        for node, dep in DG.dependencies.items():
            node_mock = module_to_node_mock[node.module]
            assert node.module == node_mock.module
            assert len(dep) == len(dependencies_mock[node_mock])
            for dep_node, dep_node_mock in zip(dep, dependencies_mock[node_mock]):
                assert dep_node.module == dep_node_mock.module
