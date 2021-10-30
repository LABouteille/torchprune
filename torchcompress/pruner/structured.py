from __future__ import annotations

from typing import List

import torch.nn as nn


def prune_conv_in(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.in_channels)) - set(indices))
    layer.in_channels = layer.in_channels - len(indices)
    layer.weight = nn.Parameter(layer.weight.data.clone()[keep_indices])


def prune_conv_out(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.out_channels)) - set(indices))
    layer.out_channels = layer.out_channels - len(indices)
    layer.weight = nn.Parameter(layer.weight.data.clone()[keep_indices])


def prune_activation_in(layer: nn.Module, indices: "List[int]") -> None:
    pass


def prune_activation_out(layer: nn.Module, indices: "List[int]") -> None:
    pass
