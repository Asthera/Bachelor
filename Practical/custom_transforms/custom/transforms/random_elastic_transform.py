"""Just reimplementation of torchvision.transforms.ElasticTransform with random apply and random parameters"""
import torch
import numpy as np
import torchvision.transforms as T


class RandomElasticTransform(object):
    def __init__(self, p=0.5, alpha=(5.0, 50.0), sigma=(3.0, 10.0)):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if torch.rand(1) < self.p:
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            return T.ElasticTransform(alpha=alpha, sigma=sigma)(img)
        return img
