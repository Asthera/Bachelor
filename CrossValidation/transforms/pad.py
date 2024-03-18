from torchvision.transforms import Pad


class TransformPad:
    def __init__(self, square_size):
        self.square_size = square_size
        self.base_pad = Pad(0, padding_mode="edge")

    def __call__(self, img):
        *rest, h, w = img.shape

        if h > self.square_size or w > self.square_size:
            raise ValueError("Image sizes must be smaller or equal to max image size!")

        left_pad = (self.square_size - w) // 2
        right_pad = left_pad if w + 2 * left_pad == self.square_size else left_pad + 1
        top_pad = (self.square_size - h) // 2
        bottom_pad = top_pad if h + 2 * top_pad == self.square_size else top_pad + 1

        self.base_pad.padding = (left_pad, top_pad, right_pad, bottom_pad)

        return self.base_pad(img)
