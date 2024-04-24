import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT

transform_0 = [
    None
]

# Translation
transform_1_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
]

transform_1_1 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
]

