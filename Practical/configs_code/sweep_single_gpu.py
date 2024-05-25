import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_1 = [
    None
]

# Define the transformations
transform_2 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
]

transform_3 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
]

transform_4 = [
    T.RandomApply([T.RandomAffine(degrees=(-25, 25)) ],  p=0.5),
]

transform_5 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.1, 0.5)),
]

transform_6 = [
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.01, 0.03)),
]

transform_7 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.002),
]

transform_8 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.1),
]

transform_9 = [
    T.RandomApply([MT.SavitzkyGolaySmooth(window_length=5, order=2)], p=0.5),
]

transform_10 = [
    T.RandomApply([MT.SavitzkyGolaySmooth(window_length=3, order=1)], p=0.5),
]

transform_11 = [
    T.RandomVerticalFlip(p=0.5),
]

transform_12 = [
    T.RandomHorizontalFlip(p=0.5),
]

transform_13 = [
    MT.RandZoom(prob=0.5, min_zoom=1.0, max_zoom=1.2),
]

transform_14 = [
    MT.RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.0),
]

transform_15 = [
    T.RandomResizedCrop(size=(600, 600), scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True),
]

transform_16 = [
    T.RandomResizedCrop(size=(600, 600), scale=(0.7, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True),
]

transform_17 = [
    T.RandomResizedCrop(size=(600, 600), scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True),
]

transform_18 = [
    T.RandomApply([T.ColorJitter(contrast=(0.5, 1.5))], p=0.5),
]

transform_19 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 600), rand_size=(5, 5)),
]

transform_20 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 600), rand_size=(2, 2)),
]

transform_21 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 600), rand_size=(10, 10)),
]

transform_22 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.75, 0.75)),
]

transform_23 = [
    T.RandomApply([T.ColorJitter(brightness=(0.5, 1.5))], p=0.5),
]

transform_24 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(0.15, 0.25)),
]

transform_25 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(0.05, 0.15)),
]

transform_26 = [
    T.RandomApply([T.ColorJitter(saturation=0.5)], p=0.5),
]

transform_27 = [
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
]

transform_28 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.001),
]

transform_29 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.01),
]

transform_30 = [
    MT.RandGaussianSharpen(prob=0.5, alpha=(1.0, 5.0)),
]

transform_31 = [
    MT.RandGaussianSharpen(prob=0.5),
]

transform_32 = [
    T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
]

transform_33 = [
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
]

transform_34 = [
    T.RandomAdjustSharpness(sharpness_factor=2.5, p=0.5),
]

transform_35 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(-0.1, 0.1)),
]

transform_36 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(-0.05, 0.05)),
]

transform_37 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.1)),
]

transform_38 = [
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
]

transform_39 = [
    T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False),
]
