from __future__ import annotations

import random
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    # Not import during run-time.
    import torch.nn as nn


def random_criteria(layer: nn.Module, amount_to_prune: float) -> List[int]:
    if amount_to_prune <= 0.0:
        return []
    n = len(layer.weight)
    if amount_to_prune > 1.0:
        raise ValueError(
            f"random_strategy: amount_to_prune should be [0., 1.]. Current input value = {amount_to_prune}"
        )
    n_to_prune = 1 if (int(amount_to_prune * n) == 0) else int(amount_to_prune * n)
    indices = random.sample(list(range(n)), k=n_to_prune)
    return indices
