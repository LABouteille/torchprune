import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock

import torchcompress as tc
from torchcompress.dependency_graph import OPTYPE, Node


class TestDependencyGraph:
    @classmethod
    def setup_class(cls):
        cls._seed_everything(42)

        cls.model_sequential = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1),
        )

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3)
                self.conv2 = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                return x

        cls.model_class = ConvNet()

    @classmethod
    def teardown_class(cls):
        del cls.model_sequential
        del cls.model_class

    @classmethod
    def _seed_everything(cls, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def test_build_graph_sequential(self):
        x = torch.randn(1, 2, 4, 4)
        DG = tc.DependencyGraph()
        graph = DG._DependencyGraph__build_graph(self.model_sequential, x)

        graph_mock = Mock()
        graph_copy = graph.copy()

        def getitem(module):
            return graph_copy[module]

        def setitem(module, node):
            graph_copy[module] = node

        graph_mock.__getitem__ = Mock(side_effect=getitem)
        graph_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model_sequential.modules())
        conv_2_4_module = list_modules[1]
        relu_module = list_modules[2]
        conv_4_5_module = list_modules[3]

        graph_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        graph_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        graph_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        for module, node in graph.items():
            node_mock = graph_mock[module]
            assert node.module == node_mock.module
            assert graph[module].prune_fn_next() == graph_mock[module].prune_fn_next()

    def test_build_dependency_graph_sequential(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph()
        graph = DG.build_dependency_graph(self.model_sequential, x)

        # Mock graph
        graph_mock = Mock()
        graph_copy = graph.copy()

        def getitem(module):
            return graph_copy[module]

        def setitem(module, node):
            graph_copy[module] = node

        graph_mock.__getitem__ = Mock(side_effect=getitem)
        graph_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model_sequential.modules())
        conv_2_4_module = list_modules[1]
        relu_module = list_modules[2]
        conv_4_5_module = list_modules[3]

        graph_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        graph_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        graph_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        graph_mock[conv_2_4_module].prune_fn_next = lambda: "prune_activation"
        graph_mock[relu_module].prune_fn_next = lambda: "prune_conv"
        graph_mock[conv_4_5_module].prune_fn_next = lambda: None

        graph_mock[conv_2_4_module].dependencies = [
            (graph_mock[relu_module], graph_mock[conv_2_4_module].prune_fn_next)
        ]
        graph_mock[relu_module].dependencies = [
            (graph_mock[conv_4_5_module], graph_mock[relu_module].prune_fn_next)
        ]
        graph_mock[conv_4_5_module].dependencies = []

        for module, node in graph.items():
            node_mock = graph_mock[module]
            assert node.module == node_mock.module
            assert graph[module].prune_fn_next() == graph_mock[module].prune_fn_next()

            for i, (node_dep, prune_fn_next_dep) in enumerate(
                graph[module].dependencies
            ):
                node_dep_mock, prune_fn_next_dep_mock = graph_mock[module].dependencies[
                    i
                ]
                assert node_dep.module == node_dep_mock.module
                assert prune_fn_next_dep() == prune_fn_next_dep_mock()

    def test_build_graph_class(self):
        x = torch.randn(1, 2, 4, 4)
        DG = tc.DependencyGraph()
        graph = DG._DependencyGraph__build_graph(self.model_class, x)

        graph_mock = Mock()
        graph_copy = graph.copy()

        def getitem(module):
            return graph_copy[module]

        def setitem(module, node):
            graph_copy[module] = node

        graph_mock.__getitem__ = Mock(side_effect=getitem)
        graph_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model_class.modules())
        conv_2_4_module = list_modules[1]
        relu_module = OPTYPE.FUNCTIONAL
        conv_4_5_module = list_modules[2]

        graph_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        graph_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        graph_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        for module, node in graph.items():
            node_mock = graph_mock[module]
            assert node.module == node_mock.module
            assert graph[module].prune_fn_next() == graph_mock[module].prune_fn_next()

    def test_build_dependency_graph_class(self):
        x = torch.randn(1, 2, 4, 4)

        DG = tc.DependencyGraph()
        graph = DG.build_dependency_graph(self.model_class, x)

        # Mock graph
        graph_mock = Mock()
        graph_copy = graph.copy()

        def getitem(module):
            return graph_copy[module]

        def setitem(module, node):
            graph_copy[module] = node

        graph_mock.__getitem__ = Mock(side_effect=getitem)
        graph_mock.__setitem__ = Mock(side_effect=setitem)

        list_modules = list(self.model_class.modules())
        conv_2_4_module = list_modules[1]
        relu_module = OPTYPE.FUNCTIONAL
        conv_4_5_module = list_modules[2]

        graph_mock[conv_2_4_module] = Node(
            module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )
        graph_mock[relu_module] = Node(
            module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
        )
        graph_mock[conv_4_5_module] = Node(
            module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
        )

        graph_mock[conv_2_4_module].prune_fn_next = lambda: "prune_activation"
        graph_mock[relu_module].prune_fn_next = lambda: "prune_conv"
        graph_mock[conv_4_5_module].prune_fn_next = lambda: None

        graph_mock[conv_2_4_module].dependencies = [
            (graph_mock[relu_module], graph_mock[conv_2_4_module].prune_fn_next)
        ]
        graph_mock[relu_module].dependencies = [
            (graph_mock[conv_4_5_module], graph_mock[relu_module].prune_fn_next)
        ]
        graph_mock[conv_4_5_module].dependencies = []

        for module, node in graph.items():
            node_mock = graph_mock[module]
            assert node.module == node_mock.module
            assert graph[module].prune_fn_next() == graph_mock[module].prune_fn_next()

            for i, (node_dep, prune_fn_next_dep) in enumerate(
                graph[module].dependencies
            ):
                node_dep_mock, prune_fn_next_dep_mock = graph_mock[module].dependencies[
                    i
                ]
                assert node_dep.module == node_dep_mock.module
                assert prune_fn_next_dep() == prune_fn_next_dep_mock()
