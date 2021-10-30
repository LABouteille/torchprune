from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    # Not import during run-time.
    import torch.nn as nn


def random_strategy(layer: nn.Module, amount_to_prune: float) -> List[int]:
    if amount_to_prune <= 0:
        return []
    n = len(layer.weight)
    n_to_prune = (
        int(amount_to_prune * n) if amount_to_prune < 1.0 else int(amount_to_prune)
    )
    if n_to_prune == 0:
        return []
    indices = random.sample(list(range(n)), k=n_to_prune)
    return indices
