from typing import Tuple
from torchvision import transforms


class TransformResize:
    def __init__(self, size: int | Tuple[int, int]):
        self.target_h = size if isinstance(size, int) else size[0]
        self.target_w = size if isinstance(size, int) else size[1]

        self.base_resize = transforms.Resize(size, antialias=True)

    def __call__(self, img):
        *rest, img_h, img_w = img.shape

        if (self.target_h * img_w) / (self.target_w * img_h) < 1:
            new_h = self.target_h
            new_w = int((self.target_h * img_w) / img_h)
        else:
            new_h = int((self.target_w * img_h) / img_w)
            new_w = self.target_w

        self.base_resize.size = (new_h, new_w)

        return self.base_resize(img)
