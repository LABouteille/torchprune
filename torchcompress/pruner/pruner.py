from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List

import torch.nn as nn

from torchcompress.pruner.node import OPTYPE

if TYPE_CHECKING:
    # Not import during run-time.
    import torch

    from torchcompress.pruner.dependency_graph import DependencyGraph
    from torchcompress.pruner.node import Node


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
            if dep.op_type == OPTYPE.FLATTEN:
                indices = self.__expand_indices(input_node, indices)
            dep.prune_fn["in_channels"](dep.module, indices)

    def __expand_indices(self, input_node: Node, indices: List[int]):
        """"""
        x, i = self.dummy_input, 0

        # Compute output for each convolution/linear layer.
        do_while = i < len(self.ordered_node)
        while do_while:

            if self.ordered_node[i].op_type != OPTYPE.ACTIVATION:
                x = self.ordered_node[i].module(x)

            do_while = i < len(self.ordered_node) and self.ordered_node[i] != input_node

            i += 1

        _, _, h, w = x.shape

        new_indices: List[int] = []

        for idx in indices:
            tmp = [idx * (h * w)]
            for j in range(1, h * w):
                tmp.append(tmp[j - 1] + 1)
            new_indices += tmp

        return new_indices
