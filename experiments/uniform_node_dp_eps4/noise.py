"""Full2340-coordinate sigma139 approximations, matching original source design."""
import random
import torch


class SecretGaussian:
    """Future research source; U2 normal Sessions never reach secret draws."""
    def __init__(self):self.__rng=random.SystemRandom();self.calls=0

    def draw(self):
        self.calls+=1
        return torch.tensor([self.__rng.normalvariate(0.,139.) for _ in range(2340)],dtype=torch.float64)


class TestGaussian:
    """Public artificial-test source; never accepted as a real research source."""
    def __init__(self,seed):self.__rng=random.Random(seed);self.calls=0

    def draw(self):
        self.calls+=1
        return torch.tensor([self.__rng.normalvariate(0.,139.) for _ in range(2340)],dtype=torch.float64)
