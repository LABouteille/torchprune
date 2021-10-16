import torch
from typing import Any, Dict


class Node:
    def __init__(self, module, grad_fn):
        self.module = module
        self.grad_fn = grad_fn
        self.parents = []

    def __repr__(self):
        return f"<Node: {self.grad_fn}>"


class DependencyGraph:
    def __init__(self):
        pass

    def build_dependency_graph(self, model: torch.nn.Module, inputs: torch.Tensor):
        """"""
        # Build graph
        self.graph_ = self.__build_graph(model, inputs)
        # Build dependency
        self.dependency_graph = self.__build_dependency(self.graph_)

    def __build_graph(self, model: torch.nn.Module, inputs: torch.Tensor):
        """"""
        grad_fn_to_module = {}

        # Register hooks
        def hook_fn(module, input, output):
            """"""
            grad_fn_to_module[output.grad_fn] = module

        hooks = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))

        out = model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Build graph
        module_to_node = {}

        def backward_traversal(grad_fn: Any):
            """"""
            module = grad_fn_to_module.get(grad_fn, None)

            if module is None:
                module = grad_fn

            node = Node(module, grad_fn)
            module_to_node[module] = node

            if hasattr(grad_fn, "next_functions"):
                for parent in grad_fn.next_functions:
                    if parent[0] is not None:
                        if (
                            "accumulategrad" not in parent[0].name().lower()
                        ):  # Ignore leaf nodes.
                            node.parents.append(parent[0])
                            backward_traversal(parent[0])

        backward_traversal(out.grad_fn)
        return module_to_node

    def __build_dependency(self, graph: Dict[Any, torch.nn.Module]):
        pass
