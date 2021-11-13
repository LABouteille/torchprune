from __future__ import annotations

from typing import List

import torch.nn as nn


def prune_conv_in(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.in_channels)) - set(indices))
    layer.in_channels = layer.in_channels - len(indices)
    layer.weight = nn.Parameter(layer.weight.data.clone()[:, keep_indices])


def prune_conv_out(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.out_channels)) - set(indices))
    layer.out_channels = layer.out_channels - len(indices)
    layer.weight = nn.Parameter(layer.weight.data.clone()[keep_indices])
    if layer.bias is not None:  # No need bias if Batchnorm is activated
        layer.bias = nn.Parameter(layer.bias.data.clone()[keep_indices])


def prune_linear_in(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.in_features)) - set(indices))
    layer.in_features = layer.in_features - len(indices)
    layer.weight = nn.Parameter(layer.weight.data.clone()[:, keep_indices])


def prune_linear_out(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.out_features)) - set(indices))
    layer.out_features = layer.out_features - len(indices)
    layer.weight = nn.Parameter(layer.weight.data.clone()[keep_indices])
    if layer.bias is not None:  # No need bias if Batchnorm is activated
        layer.bias = nn.Parameter(layer.bias.data.clone()[keep_indices])


def prune_batchnorm_in(layer: nn.Module, indices: "List[int]") -> None:
    keep_indices = list(set(range(layer.num_features)) - set(indices))
    layer.num_features = layer.num_features - len(indices)
    layer.running_mean = layer.running_mean.data.clone()[keep_indices]
    layer.running_var = layer.running_var.data.clone()[keep_indices]

    if layer.affine:
        layer.weight = nn.Parameter(layer.weight.data.clone()[keep_indices])
        layer.bias = nn.Parameter(layer.bias.data.clone()[keep_indices])


def prune_batchnorm_out(layer: nn.Module, indices: "List[int]") -> None:
    pass
