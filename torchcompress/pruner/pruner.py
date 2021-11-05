from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List

from torchcompress.node import OPTYPE

if TYPE_CHECKING:
    # Not import during run-time.
    import torch
    import torch.nn as nn

    from torchcompress.dependency_graph import DependencyGraph
    from torchcompress.node import Node


class Pruner:
    def __init__(self, DG: DependencyGraph, dummy_input: torch.Tensor):
        self.dummy_input = dummy_input
        self.module_to_node: Dict[nn.Module, Node] = DG.module_to_node
        self.ordered_node: List[Node] = DG.ordered_node
        self.dependencies: Dict[Node, List[Node]] = DG.dependencies

    def run(self, layer: nn.Module, criteria: Callable, amount_to_prune: float) -> None:
        """"""
        indices: List[int] = criteria(layer, amount_to_prune)

        input_node = self.module_to_node[layer]
        input_node.prune_fn["out_channels"](input_node.module, indices)

        for dep in self.dependencies[input_node]:
            if dep.op_type == OPTYPE.RESHAPE:
                # TODO: Use some sort of self.cache to avoid doing module(x) all over again ?
                self.__reshape_indices(input_node, indices)
                break

        for dep in self.dependencies[input_node]:
            dep.prune_fn["in_channels"](dep.module, indices)

    def __reshape_indices(self, input_node, indices):
        """"""
        x = self.dummy_input
        i = 0

        while i < len(self.ordered_node) and self.ordered_node[i] != input_node:
            if self.ordered_node[i].op_type != OPTYPE.ACTIVATION:
                x = self.ordered_node[i].module(x)
            i += 1

        n = x.shape[0]
        c = input_node.module.out_channels
        h = x.shape[2] - input_node.module.kernel_size[0] + 1
        w = x.shape[3] - input_node.module.kernel_size[1] + 1

        new_flatten_size = n * c * h * w
        # TODO: How to reshape indices ?

        print(new_flatten_size)

        import pdb

        pdb.set_trace()
