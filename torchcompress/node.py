import torch.nn as nn
from enum import Enum
from typing import Any, Callable, List, Tuple


class OPTYPE(Enum):
    CONV = 1
    ACTIVATION = 2
    # For torch.nn.functional
    FUNCTIONAL = 3

    def __repr__(self):
        return self.name


class Node:
    def __init__(self, module: nn.Module, op_type: OPTYPE, grad_fn: Any):
        # Public
        self.module: nn.Module = module
        self.op_type: OPTYPE = op_type
        self.grad_fn = grad_fn
        self.prune_fn_next: Callable = lambda: None
        self.dependencies: List[Tuple[nn.Module, Callable]] = []
        self.outputs: List[nn.Module] = []

    def __repr__(self):
        return f"<Node: module={self.module} | op_type={self.op_type} | prune_fn_next={self.prune_fn_next()}>"
