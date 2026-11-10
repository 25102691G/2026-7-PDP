"""1283-coordinate sources. C2 executes only the explicit TestGaussian path."""
import random
import torch


class SecretGaussian:
    def __init__(self):self.__rng=random.SystemRandom();self.calls=0
    def draw(self):
        self.calls+=1
        return torch.tensor([self.__rng.normalvariate(0.,260.) for _ in range(1283)],dtype=torch.float64)


class TestGaussian:
    def __init__(self,seed):self.__rng=random.Random(seed);self.calls=0
    def draw(self):
        self.calls+=1
        return torch.tensor([self.__rng.normalvariate(0.,260.) for _ in range(1283)],dtype=torch.float64)
