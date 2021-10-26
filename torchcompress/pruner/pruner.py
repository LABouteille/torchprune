from typing import List

from torchcompress.node import OPTYPE, Node
from torchcompress.pruner.criteria import random_strategy
from torchcompress.pruner.structured import prune_conv


class Pruner:
    def __init__(self, ordered_node: List[Node]):

        input_node = Node(
            ordered_node[0].module, ordered_node[0].op_type, ordered_node[0].grad_fn
        )
        if input_node.op_type == OPTYPE.CONV:
            input_node.prune_fn_next = lambda: prune_conv(input_node)

        self.ordered_node = [input_node] + ordered_node

    def run(self):
        for node in self.ordered_node:
            print(node.module)
            # 1) Calculate number of node to prune (criteria)
            print(f"\t{random_strategy()}")
            # 2) Run pruning
            print(f"\t{node.prune_fn_next()}")
