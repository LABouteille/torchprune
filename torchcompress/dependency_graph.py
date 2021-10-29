import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:  # Not import during run-time.
    import torch

from torchcompress.node import OPTYPE, Node
from torchcompress.pruner.structured import (
    prune_activation_out,
    prune_conv_in,
    prune_conv_out,
)


class DependencyGraph:
    def __init__(self, model: "nn.Module"):
        self.model = model

    def build_dependency_graph(self, inputs: "torch.Tensor"):
        """"""
        self.module_to_node: "Dict[nn.Module, Node]" = self.__build_graph(inputs)
        ordered_node: "List[Node]" = self.__order_dependency_graph(self.module_to_node)
        self.dependencies: "Dict[Node, List[Node]]" = self.__build_dependency(
            ordered_node
        )

    def __order_dependency_graph(self, module_to_node: "Dict[nn.Module, Node]"):
        """"""

        def __topological_sort(
            node: "Node", ordered_node: "List[Node]", visited: "Set[Node]"
        ):
            """"""
            if node not in visited:
                visited.add(node)
                for out in node.outputs:
                    __topological_sort(out, ordered_node, visited)
                ordered_node.append(node)
            return ordered_node

        ordered_node: "List[Node]" = []
        visited: "Set[Node]" = set()

        input_module = list(self.model.modules())[1]
        __topological_sort(module_to_node[input_module], ordered_node, visited)

        return list(reversed(ordered_node))

    def __build_graph(self, inputs: "torch.Tensor"):
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
        def __backward_traversal(
            grad_fn: "Any", module_to_node: "Dict[nn.Module, Node]"
        ):
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

        module_to_node: "Dict[nn.Module, Node]" = {}
        _ = __backward_traversal(out.grad_fn, module_to_node)
        return module_to_node

    def __build_dependency(self, ordered_node: "List[Node]"):
        """"""
        dependencies: "Dict[Node, List[Node]]" = {}

        for node in ordered_node:
            if node.op_type == OPTYPE.CONV:

                node.prune_fn["in_channels"] = prune_conv_in
                dependencies[node] = []

                # TODO Suppose len(node.outputs) == 1. What if len > 2
                out = node.outputs

                while len(out) > 0 and out[0].op_type != OPTYPE.CONV:
                    # FIXME prune_fn depends on type of node
                    out[0].prune_fn["out_channels"] = prune_activation_out
                    dependencies[node].append(out[0])
                    out = out[0].outputs

                if len(out) > 0:
                    out[0].prune_fn["out_channels"] = prune_conv_out
                    dependencies[node].append(out[0])

        return dependencies
