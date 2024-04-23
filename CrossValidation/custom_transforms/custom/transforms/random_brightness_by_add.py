import torch
import numpy as np


class RandomBrightnessByAdd(object):
    def __init__(self, p=0.5, delta=(0.2, 0.2)):
        self.p = p
        self.delta = delta

    def __call__(self, image):
        if torch.rand(1) < self.p:
            delta = np.random.uniform(self.delta[0], self.delta[1])
            return image + delta

        return image