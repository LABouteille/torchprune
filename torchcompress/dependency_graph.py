import torch
import torch.nn as nn
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple


class OPTYPE(Enum):
    CONV = 1
    ACTIVATION = 2
    # For torch.nn.functional
    FUNCTIONAL = 3

    def __repr__(self):
        return self.name


class Node:
    def __init__(self, module: nn.Module, op_type: OPTYPE, grad_fn: Any):
        # Public
        self.module: nn.Module = module
        self.op_type: OPTYPE = op_type
        self.grad_fn = grad_fn
        self.prune_fn_next: Callable = lambda: None
        self.dependencies: List[Tuple[nn.Module, Callable]] = []
        self.outputs: List[nn.Module] = []

    def __repr__(self):
        return f"<Node: module={self.module} | op_type={self.op_type} | prune_fn_next={self.prune_fn_next()}>"


class DependencyGraph:
    def __init__(self):
        pass

    def build_dependency_graph(self, model: nn.Module, inputs: torch.Tensor):
        """"""
        graph = self.__build_graph(model, inputs)
        graph = self.__build_dependency(graph)
        return graph

    def __build_graph(self, model: nn.Module, inputs: torch.Tensor):
        """"""
        grad_fn_to_module = {}

        # Register hooks
        def hook_fn(module, input, output):
            """"""
            grad_fn_to_module[output.grad_fn] = module

        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn))

        out = model(inputs)

        for hook in hooks:
            hook.remove()

        # Backward traversal
        def __backward_traversal(grad_fn: Any, module_to_node: Dict[nn.Module, Node]):
            """"""
            module = grad_fn_to_module.get(grad_fn)

            if isinstance(module, nn.Conv2d):
                op_type = OPTYPE.CONV
            else:
                # Pytorch functional has no module
                if module is None:
                    module = OPTYPE.FUNCTIONAL
                op_type = OPTYPE.ACTIVATION

            node = Node(module, op_type, grad_fn)
            module_to_node[module] = node

            if hasattr(grad_fn, "next_functions"):
                for parent in grad_fn.next_functions:
                    if parent[0] is not None:
                        if (
                            "accumulategrad" not in parent[0].name().lower()
                        ):  # Ignore leaf nodes.
                            out_node = __backward_traversal(parent[0], module_to_node)
                            out_node.outputs.append(node)
            return node

        module_to_node: Dict[nn.Module, Node] = {}
        _ = __backward_traversal(out.grad_fn, module_to_node)
        return module_to_node

    def __build_dependency(self, graph: Dict[Any, nn.Module]):
        """"""

        def prune_conv():
            return "prune_conv"

        def prune_activation():
            return "prune_activation"

        for module, node in graph.items():

            for output in node.outputs:
                if output.op_type == OPTYPE.ACTIVATION:
                    node.prune_fn_next = lambda: prune_activation()
                elif output.op_type == OPTYPE.CONV:
                    node.prune_fn_next = lambda: prune_conv()

                node.dependencies.append((output, node.prune_fn_next))

        return graph
