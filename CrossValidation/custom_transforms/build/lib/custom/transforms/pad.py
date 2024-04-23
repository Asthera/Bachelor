from typing import Tuple
from torchvision import transforms


class TransformPad:
    def __init__(self, rectangle_size: int | Tuple[int, int]):
        self.target_h = rectangle_size if isinstance(rectangle_size, int) else rectangle_size[0]
        self.target_w = rectangle_size if isinstance(rectangle_size, int) else rectangle_size[1]

        self.base_pad = transforms.Pad(0, padding_mode="edge")

    def __call__(self, img):
        *rest, img_h, img_w = img.shape

        if img_h > self.target_h or img_w > self.target_w:
            raise ValueError("Image sizes must be smaller or equal to max image size!")

        left_pad = (self.target_w - img_w) // 2
        right_pad = left_pad if img_w + 2 * left_pad == self.target_w else left_pad + 1
        top_pad = (self.target_h - img_h) // 2
        bottom_pad = top_pad if img_h + 2 * top_pad == self.target_h else top_pad + 1

        self.base_pad.padding = (left_pad, top_pad, right_pad, bottom_pad)

        return self.base_pad(img)
