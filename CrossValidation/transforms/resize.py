from torchvision.transforms import Resize


class TransformResize:
    def __init__(self, size):
        self.size = size
        self.base_resize = Resize(size, antialias=True)

    def __call__(self, img):
        *rest, h, w = img.shape

        if h > w:
            new_h = self.size
            new_w = int((self.size * w) / h)
        else:
            new_h = int((self.size * h) / w)
            new_w = self.size

        self.base_resize.size = (new_h, new_w)

        return self.base_resize(img)
