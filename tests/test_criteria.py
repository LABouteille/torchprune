import os
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

import torchcompress as tc


class TestCriteria:
    @classmethod
    def _seed_everything(cls, seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    @pytest.fixture(autouse=True, scope="function")
    def setup_at_each_test(self):
        self._seed_everything(42)

    def test_random_criteria_amount_to_prune_eq_zero(self):
        layer = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
        indices = tc.random_criteria(layer=layer, amount_to_prune=0.0)
        assert len(indices) == 0

    def test_random_criteria_amount_to_prune_le_zero(self):
        layer = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
        indices = tc.random_criteria(layer=layer, amount_to_prune=-1)
        assert len(indices) == 0

    def test_random_criteria_amount_to_prune_ge_than_one(self):
        layer = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
        with pytest.raises(ValueError):
            _ = tc.random_criteria(layer=layer, amount_to_prune=42.0)

    def test_random_criteria_n_to_prune_eq_one(self):
        layer = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
        indices = tc.random_criteria(layer=layer, amount_to_prune=0.1)
        assert len(indices) == 1

    def test_random_criteria_n_to_prune_ge_one(self):
        layer = nn.Conv2d(in_channels=4, out_channels=5, kernel_size=1)
        indices = tc.random_criteria(layer=layer, amount_to_prune=0.25)
        assert len(indices) == 1

    def test_random_criteria_n_to_prune_eq_n(self):
        layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        indices = tc.random_criteria(layer=layer, amount_to_prune=1.0)
        assert len(indices) == 1
