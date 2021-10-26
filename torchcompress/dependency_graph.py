import torch
import torch.nn as nn
from typing import Any, Dict

from torchcompress.node import OPTYPE, Node
from torchcompress.pruner.structured import prune_activation, prune_conv


class DependencyGraph:
    def __init__(self, model: nn.Module):
        self.model = model

    def build_dependency_graph(self, inputs: torch.Tensor):
        """"""
        graph: Dict[nn.Module, Node] = self.__build_graph(inputs)
        graph = self.__build_dependency(graph)
        return graph

    def __build_graph(self, inputs: torch.Tensor):
        """"""
        grad_fn_to_module = {}

        # Register hooks
        def hook_fn(module, input, output):
            """"""
            grad_fn_to_module[output.grad_fn] = module

        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn))

        out = self.model(inputs)

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

    def __build_dependency(self, graph: Dict[nn.Module, Node]):
        """"""
        for module, node in graph.items():

            for output in node.outputs:
                if output.op_type == OPTYPE.ACTIVATION:
                    node.prune_fn_next = lambda: prune_activation(output)
                elif output.op_type == OPTYPE.CONV:
                    node.prune_fn_next = lambda: prune_conv(output)

                node.dependencies.append((output, node.prune_fn_next))

        return graph
