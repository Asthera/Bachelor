import monai.transforms as MT
import torchvision.transforms as T

from .random_noise import RandomNoise
from .resize import TransformResize
from .pad import TransformPad
from .random_noise_with_fv import RandomNoiseWithFV
from .salt_and_pepper_noise import RandomSaltAndPepperNoise
from .contrast_by_multiply import RandomContrastByMultiply
from .brightness_by_add import RandomBrightnessByAdd

class TransformsBuilder:
    def __init__(self, transforms: list[str] or str) -> None:
        self.transforms = transforms
        self.compose = []

    def add(self, transform: str):

        if transform == 'None' or transform == 'none':
            self.compose.append(None)
            return

        try:
            # trying with torchvision
            transform = eval(f'T.{transform}')

        except:

            try:
                # trying with monai
                transform = eval(f'MT.{transform}')

            except:

                try:
                    # trying with custom transforms
                    transform = eval(f'{transform}')

                except:

                    raise ValueError(f"Transform {transform} not found in torchvision or monai, or custom transforms.")

        self.compose.append(transform)

    def build(self) -> T.Compose or None:

        print(self.transforms)

        if isinstance(self.transforms, str):
            self.transforms = [self.transforms]

        for transform in self.transforms:
            self.add(transform)

        if self.compose[0] is None:
            return None

        return T.Compose(self.compose)
