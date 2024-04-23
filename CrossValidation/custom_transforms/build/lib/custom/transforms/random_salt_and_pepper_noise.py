import torch
import numpy as np


class RandomSaltAndPepperNoise(object):
    def __init__(self, p=0.5, density=(0.01, 0.03)):
        self.p = p
        self.density = density

    def __call__(self, image):
        if torch.rand(1) < self.p:
            density = np.random.uniform(self.density[0], self.density[1])

            image_copy = image.clone()

            # Generate random matrix
            rand_matrix = torch.rand(image.size())

            # Create salt & pepper mask
            salt_mask = (rand_matrix < (density / 2))
            pepper_mask = (rand_matrix > 1 - (density / 2))

            # Apply masks to image
            image_copy[salt_mask] = 1
            image_copy[pepper_mask] = 0

            return image_copy

        return image
