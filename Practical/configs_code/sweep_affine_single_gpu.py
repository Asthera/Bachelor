import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT

transform_0 = [
    None
]

# Translation
transform_1_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.0))], p=0.5),
]

transform_1_1 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.0, 0.15))], p=0.5),
]

transform_1_2 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
]

transform_1_3 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.30, 0.30))], p=0.5),
]

# Rotation
transform_2_0 = [
    T.RandomApply([T.RandomAffine(degrees=(-15, 15))], p=0.5),
]

transform_2_1 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
]

transform_2_2 = [
    T.RandomApply([T.RandomAffine(degrees=(-45, 45))], p=0.5),
]

transform_2_3 = [
    T.RandomApply([T.RandomAffine(degrees=(-5, 5))], p=0.5),
]

# Shear
transform_3_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, 0, 0))], p=0.5),
]

transform_3_1 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(0, 0, -15, 15))], p=0.5),
]

transform_3_2 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5),
]

transform_3_3 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-30, 30, -30, 30))], p=0.5),
]

# Reflection
transform_4_1 = [
    T.RandomHorizontalFlip(p=0.5),
]

transform_4_2 = [
    T.RandomVerticalFlip(p=0.5),
]


# Scale, without change size
transform_5_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, scale=(1.0, 1.15))], p=0.5),
]

transform_5_1 = [
    T.RandomApply([T.RandomAffine(degrees=0, scale=(0.85, 1.0))], p=0.5),
]

transform_5_2 = [
    T.RandomApply([T.RandomAffine(degrees=0, scale=(0.85, 1.15))], p=0.5),
]

transform_5_3 = [
    T.RandomApply([T.RandomAffine(degrees=0, scale=(0.7, 1.3))], p=0.5),
]

# Scale, by random crop

transform_5_4 = [
    T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(0.7, 1.3), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5),
]

transform_5_5 = [
    T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(0.85, 1.15), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5),
]

transform_5_6 = [
    T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(0.85, 1.0), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5),
]

transform_5_7 = [
    T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(1.0, 1.15), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5),
]
