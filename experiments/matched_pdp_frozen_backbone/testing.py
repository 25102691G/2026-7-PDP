"""Explicit artificial-test RNG; cannot enter the normal scientific Session."""
import random
import torch


class TestGaussian:
    __test__ = False

    def __init__(self, seed):
        self.__rng = random.Random(seed)
        self.calls = 0

    def draw(self):
        self.calls += 1
        return torch.tensor([self.__rng.normalvariate(0.,260.) for _ in range(33)], dtype=torch.float64)
