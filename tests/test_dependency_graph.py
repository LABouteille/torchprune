import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchcompress as tc

# from unittest.mock import Mock

# from torchcompress.node import OPTYPE, Node


class TestDependencyGraph:
    @classmethod
    def setup_class(cls):
        cls._seed_everything(42)

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

        cls.model = ConvNet()

    @classmethod
    def teardown_class(cls):
        del cls.model

    @classmethod
    def _seed_everything(cls, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def test(self):
        x = torch.randn(1, 2, 4, 4)
        DG = tc.DependencyGraph(self.model)
        DG.build_dependency_graph(x)

        # for key, val in DG.dependencies.items():
        #     print(key, val)
        #     print(f"\t{key} prune_fn = {key.prune_fn['in_channels']()} | {key.prune_fn['out_channels']()}")

        #     for node in val:
        #         print(f"\t{node} prune_fn = {node.prune_fn['in_channels']()} | {node.prune_fn['out_channels']()}")
        #     print("-----")

    # def test_build_graph(self):
    #     x = torch.randn(1, 2, 4, 4)
    #     DG = tc.DependencyGraph(self.model)
    #     module_to_node = DG._DependencyGraph__build_graph(x)

    #     module_to_node_mock = Mock()
    #     module_to_node_copy = module_to_node.copy()

    #     def getitem(module):
    #         return module_to_node_copy[module]

    #     def setitem(module, node):
    #         module_to_node_copy[module] = node

    #     module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
    #     module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

    #     list_modules = list(self.model.modules())
    #     conv_2_4_module = list_modules[1]
    #     relu_module = OPTYPE.FUNCTIONAL
    #     conv_4_5_module = list_modules[2]

    #     module_to_node_mock[conv_2_4_module] = Node(
    #         module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
    #     )
    #     module_to_node_mock[relu_module] = Node(
    #         module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
    #     )
    #     module_to_node_mock[conv_4_5_module] = Node(
    #         module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
    #     )

    #     for module, node in module_to_node.items():
    #         node_mock = module_to_node_mock[module]
    #         assert node.module == node_mock.module
    #         assert module_to_node[module].prune_fn_next() == module_to_node_mock[module].prune_fn_next()

    # def test_build_dependency_graph(self):
    #     x = torch.randn(1, 2, 4, 4)

    #     DG = tc.DependencyGraph(self.model)
    #     module_to_node = DG.build_dependency_graph(x)

    #     # Mock graph
    #     module_to_node_mock = Mock()
    #     module_to_node_copy = module_to_node.copy()

    #     def getitem(module):
    #         return module_to_node_copy[module]

    #     def setitem(module, node):
    #         module_to_node_copy[module] = node

    #     module_to_node_mock.__getitem__ = Mock(side_effect=getitem)
    #     module_to_node_mock.__setitem__ = Mock(side_effect=setitem)

    #     list_modules = list(self.model.modules())
    #     conv_2_4_module = list_modules[1]
    #     relu_module = OPTYPE.FUNCTIONAL
    #     conv_4_5_module = list_modules[2]

    #     module_to_node_mock[conv_2_4_module] = Node(
    #         module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
    #     )
    #     module_to_node_mock[relu_module] = Node(
    #         module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
    #     )
    #     module_to_node_mock[conv_4_5_module] = Node(
    #         module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
    #     )

    #     module_to_node_mock[conv_2_4_module].prune_fn_next = lambda: "prune_activation"
    #     module_to_node_mock[relu_module].prune_fn_next = lambda: "prune_conv"
    #     module_to_node_mock[conv_4_5_module].prune_fn_next = lambda: None

    #     module_to_node_mock[conv_2_4_module].dependencies = [
    #         (module_to_node_mock[relu_module], module_to_node_mock[conv_2_4_module].prune_fn_next)
    #     ]
    #     module_to_node_mock[relu_module].dependencies = [
    #         (module_to_node_mock[conv_4_5_module], module_to_node_mock[relu_module].prune_fn_next)
    #     ]
    #     module_to_node_mock[conv_4_5_module].dependencies = []

    #     for module, node in module_to_node.items():
    #         node_mock = module_to_node_mock[module]
    #         assert node.module == node_mock.module
    #         assert module_to_node[module].prune_fn_next() == module_to_node_mock[module].prune_fn_next()

    #         for i, (node_dep, prune_fn_next_dep) in enumerate(
    #             module_to_node[module].dependencies
    #         ):
    #             node_dep_mock, prune_fn_next_dep_mock = module_to_node_mock[module].dependencies[
    #                 i
    #             ]
    #             assert node_dep.module == node_dep_mock.module
    #             assert prune_fn_next_dep() == prune_fn_next_dep_mock()

    # def test_order_dependency_graph(self):
    #     x = torch.randn(1, 2, 4, 4)

    #     DG = tc.DependencyGraph(self.model)
    #     module_to_node = DG.build_dependency_graph(x)
    #     ordered_node = DG.order_dependency_graph(module_to_node)

    #     list_modules = list(self.model.modules())
    #     conv_2_4_module = list_modules[1]
    #     relu_module = OPTYPE.FUNCTIONAL
    #     conv_4_5_module = list_modules[2]

    #     node1_mock = Node(
    #         module=conv_2_4_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
    #     )
    #     node2_mock = Node(
    #         module=relu_module, op_type=OPTYPE.ACTIVATION, grad_fn=lambda: None
    #     )
    #     node3_mock = Node(
    #         module=conv_4_5_module, op_type=OPTYPE.CONV, grad_fn=lambda: None
    #     )

    #     node1_mock.prune_fn_next = lambda: "prune_activation"
    #     node2_mock.prune_fn_next = lambda: "prune_conv"
    #     node3_mock.prune_fn_next = lambda: None

    #     ordered_node_mock = [node1_mock, node2_mock, node3_mock]

    #     for i, node in enumerate(ordered_node):
    #         node_mock = ordered_node_mock[i]
    #         assert node.module == node_mock.module
    #         assert node.prune_fn_next() == node_mock.prune_fn_next()
