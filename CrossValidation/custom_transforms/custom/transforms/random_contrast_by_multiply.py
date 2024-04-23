import torch
import numpy as np


class RandomContrastByMultiply(object):
    def __init__(self, p=0.5, multiplier=(0.5, 1.5)):
        self.p = p
        self.lower = multiplier[0]
        self.upper = multiplier[1]

    def __call__(self, image):
        if torch.rand(1) < self.p:
            multiplier = np.random.uniform(self.lower, self.upper)
            return image * multiplier
        return image