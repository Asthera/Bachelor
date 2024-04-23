import torch
import numpy as np


class RandomNoiseWithFV(object):
    def __init__(self, p=0.5, fv=(0.1, 0.5)):
        self.p = p
        self.fv = fv

    def __call__(self, image):
        if torch.rand(1, device=image.device) < self.p:
            fv = np.random.uniform(self.fv[0], self.fv[1])

            # Make sure to move print statements outside of production code or control them with a debug flag
            print(fv)
            variance = torch.var(image)

            # Ensure that noise is created as a Tensor directly and matches the image data type
            noise = torch.randn(image.size(), device=image.device, dtype=image.dtype) * np.sqrt(variance)

            # Adjust the random scaling factor to the correct type and device
            scale_factor = torch.rand(image.size(), device=image.device, dtype=image.dtype) - fv

            # Apply the noise scaled by the random factor
            image_noise = image + scale_factor * noise

            # Clip values to ensure they are within [0, 1]
            image_noise = torch.clamp(image_noise, 0, 1)

            return image_noise

        return image
