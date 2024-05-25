import monai.transforms as MT
import torchvision.transforms as T

transform_1 = [
    None
]

- - "RandomResizedCrop(510, antialias=True)"
- "RandomVerticalFlip()"
- "RandomRotation(degrees=15)"
- "ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)"
- "RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)"
- "Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))"
transform_2 = [
    T.RandomResizedCrop(510, antialias=True),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]

- - "RandomResizedCrop(510, antialias=True)"
- "RandomVerticalFlip()"
- "RandomRotation(degrees=15)"
- "RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)"
- "Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))"

transform_3 = [
    T.RandomResizedCrop(510, antialias=True),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]

