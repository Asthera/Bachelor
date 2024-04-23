import torch
import numpy as np

class RandomNoiseWithFV(object):
    def __init__(self, p=0.5, fv=(0.1, 0.5)):
        self.p = p
        self.fv = fv

    def __call__(self, image):
        if torch.rand(1) < self.p:
            fv = np.random.uniform(self.fv[0], self.fv[1])
            variance = torch.var(image)
            noise = np.random.normal(0, np.sqrt(variance), image.shape)
            d0, d1, d2 = image.shape
            image_noise = image + (np.random.rand(d0, d1, d2) - fv) * noise
            image_noise = torch.clamp(image_noise, 0, 1)
            return image_noise
        return image
