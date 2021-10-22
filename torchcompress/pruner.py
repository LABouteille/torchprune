from typing import List

from torchcompress.node import OPTYPE, Node


class Pruner:
    def __init__(self, ordered_node: List[Node]):

        node = Node(
            ordered_node[0].module, ordered_node[0].op_type, ordered_node[0].grad_fn
        )
        if node.op_type == OPTYPE.CONV:
            node.prune_fn_next = lambda: prune_conv(node)

        self.ordered_node = [node] + ordered_node

    def run(self):
        for node in self.ordered_node:
            print(node.module)


def prune_conv(next_node: Node):
    return "prune_conv"


def prune_activation(next_node: Node):
    return "prune_activation"
