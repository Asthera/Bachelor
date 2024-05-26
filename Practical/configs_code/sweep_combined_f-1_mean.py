import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT

# Original Combinations

transform_1 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2))
]

transform_2 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.4))
]

transform_3 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
]

transform_4 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5)
]

transform_5 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.005, 0.01))
]

# Mirrored Combinations

transform_1_mirror = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2)),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]

transform_2_mirror = [
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.4)),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]

transform_3_mirror = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]

transform_4_mirror = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]

transform_5_mirror = [
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.005, 0.01)),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]