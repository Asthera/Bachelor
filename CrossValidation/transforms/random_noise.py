from torch import rand, randn, clamp


class RandomNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if rand(1) < self.p:
            noise = randn(img.size()) * self.std + self.mean
            img_tensor = clamp(img + noise, 0, 1)

            return img_tensor
        return img
