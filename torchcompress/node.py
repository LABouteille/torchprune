from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List

if TYPE_CHECKING:  # Not import during run-time.
    import torch.nn as nn


class OPTYPE(Enum):
    CONV = 1
    ACTIVATION = 2
    # For torch.nn.functional
    FUNCTIONAL = 3

    def __repr__(self):
        return self.name


class Node:
    def __init__(self, module: "nn.Module", op_type: "OPTYPE", grad_fn: "Any"):
        # Public
        self.module: "nn.Module" = module
        self.op_type: "OPTYPE" = op_type
        self.grad_fn: "Any" = grad_fn
        self.prune_fn: "Dict[str, Callable]" = {
            "in_channels": lambda: None,
            "out_channels": lambda: None,
        }
        self.outputs: List[nn.Module] = []

    def __repr__(self):
        return f"<Node: module={self.module} | op_type={self.op_type}>"
