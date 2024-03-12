import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms as T

from utilsweep import RandomSpeckleNoise

class RandomNoise(object):
    def __init__(self, p=0.5, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if torch.rand(1) < self.p:

            noise = torch.randn(img.size()) * self.std + self.mean
            img_tensor = torch.clamp(img + noise, 0, 1)

            return img_tensor
        return img

img_path = "/Volumes/LaCie/Macbook/cropped/video_frames/001_image_211589708848.mp4/frame_19.png"

tensor = read_image(img_path)

print(tensor.shape)

transform_0 = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
transform_1 = RandomSpeckleNoise(p=1, mean=0.8, std=0.0001)
transform_2 = RandomNoise(p=1, mean=0.4, std=0.5)
transform_2 = RandomNoise(p=1, mean=0, std=0.1)


img = T.ToPILImage()(transform_2(tensor))

plt.imshow(img, cmap="gray")
plt.show()
