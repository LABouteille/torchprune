from torchcompress.node import Node


def prune_conv(next_node: Node):
    return "prune_conv"


def prune_activation(next_node: Node):
    return "prune_activation"
