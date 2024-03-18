from torch import rand, randn


class RandomSpeckleNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        super().__init__()
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if rand(1) < self.p:
            noise = randn(img.size()) * self.std + self.mean
            img = img + img * noise
        return img