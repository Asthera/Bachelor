import monai.transforms as MT
import torchvision.transforms as T
import custom.transforms as CT

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

        except NameError:
            raise ValueError(f"Transform {transform} is not defined in torchvision.transforms, monai.transforms or custom.transforms")

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
