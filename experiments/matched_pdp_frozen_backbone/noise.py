"""33D Gaussian draw interface; OS entropy, no secret seed or serialized state.

Finite-precision normalvariate is a numerical realization, not an exact Gaussian
or machine-level DP certificate. Never draw2340 then slice to33.
"""
import random
import torch


class SecretGaussian:
    def __init__(self):
        self.__rng = random.SystemRandom()
        self.calls = 0

    def draw(self):
        self.calls += 1
        return torch.tensor([self.__rng.normalvariate(0.,260.) for _ in range(33)], dtype=torch.float64)
