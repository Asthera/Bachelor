from torchvision.transforms import Compose, RandomCrop, RandomErasing, RandomHorizontalFlip, RandomRotation, \
    RandomVerticalFlip, ColorJitter
from .random_noise import RandomNoise
from .resize import TransformResize
from .pad import TransformPad


class TransformsBuilder:
    def __init__(self, transforms: list[str] or str) -> None:
        self.transforms = transforms
        self.compose = []

    def add(self, transform: str):

        transform_map = {
            "random_erasing_delete": RandomErasing(value=0),
            "random_erasing_random": RandomErasing(value="random"),
            "random_crop": RandomCrop(size=(200, 200)),
            "random_horizontal_flip": RandomHorizontalFlip(),
            "random_vertical_flip": RandomVerticalFlip(),
            "random_rotation": RandomRotation(degrees=70),
            "color_jitter": ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            "random_noise(0.4, 0.5)": RandomNoise(p=0.5, mean=0.4, std=0.5),
            "random_noise(0, 0.1)": RandomNoise(p=0.5, mean=0, std=0.1),
            "transform_resize(600)": TransformResize(600),
            "transform_resize(400)": TransformResize(400),
            "transform_pad(600)": TransformPad(600),
            "none": None
        }

        if transform not in transform_map:
            raise ValueError(f"Transform {transform} not supported, please choose one of {list(transform_map.keys())}")

        else:
            self.compose.append(transform_map[transform])

    def build(self) -> Compose or None:

        print(self.transforms)

        if isinstance(self.transforms, str):
            self.transforms = [self.transforms]

        for transform in self.transforms:
            self.add(transform)

        if self.compose[0] is None:
            return None

        return Compose(self.compose)
