from monai.transforms import RandGaussianNoise, RandAdjustContrast, RandHistogramShift, RandBiasField, RandFlip, \
    RandGibbsNoise, RandZoom, RandRotate, NormalizeIntensity
from torchvision.transforms import Compose, RandomCrop, RandomErasing, RandomHorizontalFlip, RandomRotation, \
    RandomVerticalFlip, ColorJitter, RandomResizedCrop, Normalize, RandomAffine
from .random_noise import RandomNoise
from .resize import TransformResize
from .pad import TransformPad


class TransformsBuilder:
    def __init__(self, transforms: list[str] or str) -> None:
        self.transforms = transforms
        self.compose = []

    def add(self, transform: str):

        if transform == 'None' or transform == 'none':
            self.compose.append(None)
            return

        try:
            transform = eval(transform)

        except:
            print(f"Transform {transform} not found!")
            print("Or maybe you forgot to import it?")
            print("Or maybe you have some error with name or params?")
            raise ValueError(f"Error with transform {transform}")


        self.compose.append(transform)

    def build(self) -> Compose or None:

        print(self.transforms)

        if isinstance(self.transforms, str):
            self.transforms = [self.transforms]

        for transform in self.transforms:
            self.add(transform)

        if self.compose[0] is None:
            return None

        return Compose(self.compose)
