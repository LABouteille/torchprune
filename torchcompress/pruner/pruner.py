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
        nb_filter_before_pruning = input_node.module.weight.shape[0]

        input_node.prune_fn["out_channels"](input_node.module, indices)

        for dep in self.dependencies[input_node]:
            if dep.op_type == OPTYPE.FLATTEN:
                indices = self.__expand_indices(
                    input_node, indices, nb_filter_before_pruning
                )
                break

        for dep in self.dependencies[input_node]:
            dep.prune_fn["in_channels"](dep.module, indices)

    def __expand_indices(
        self, input_node: Node, indices: List[int], nb_filter_before_pruning: int
    ):
        """"""
        new_indices: List[int] = []

        for idx in indices:
            tmp = [idx * nb_filter_before_pruning]
            for j in range(1, nb_filter_before_pruning):
                tmp.append(tmp[j - 1] + 1)
            new_indices += tmp

        return new_indices
