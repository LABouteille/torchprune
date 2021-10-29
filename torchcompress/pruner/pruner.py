from typing import TYPE_CHECKING, Callable, Dict, List

if TYPE_CHECKING:  # Not import during run-time.
    import torch.nn as nn

    from torchcompress.dependency_graph import DependencyGraph
    from torchcompress.node import Node


class Pruner:
    def __init__(self, DG: "DependencyGraph"):
        self.dependencies: "Dict[Node, List[Node]]" = DG.dependencies
        self.module_to_node: "Dict[nn.Module, Node]" = DG.module_to_node

    def run(self, layer: "nn.Module", criteria: "Callable", amount_to_prune: "float"):

        indices = criteria(layer, amount_to_prune)
        input_node = self.module_to_node[layer]

        print(f'{input_node.module} => {input_node.prune_fn["in_channels"](indices)}')

        for dep in self.dependencies[input_node]:
            print(f'\t {dep.module} => {dep.prune_fn["out_channels"](indices)}')

        print("---")
