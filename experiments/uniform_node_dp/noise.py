"""Separate fixed486 Gaussian approximations; test sources are synthetic only."""
import random
import torch


class SecretGaussian:
    def __init__(self):self.__rng=random.SystemRandom();self.calls=0

    def draw(self):
        self.calls+=1
        return torch.tensor([self.__rng.normalvariate(0.,486.) for _ in range(2340)],dtype=torch.float64)


class TestGaussian:
    def __init__(self,seed):self.__rng=random.Random(seed);self.calls=0

    def draw(self):
        self.calls+=1
        return torch.tensor([self.__rng.normalvariate(0.,486.) for _ in range(2340)],dtype=torch.float64)
