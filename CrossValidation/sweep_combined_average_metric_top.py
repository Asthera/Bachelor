import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_0 = [
    None
]

transform_1_0 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5),
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.4))
]

# just mirror for previous transform
transform_1_1 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.4)),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]

transform_2_0 = [
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5),
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.005, 0.01)),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(0.85, 1.15), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(0, 0, -15, 15))], p=0.5)
]

# just mirror for previous transform
transform_2_1 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(0, 0, -15, 15))], p=0.5),
    T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(0.85, 1.15), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.005, 0.01)),
    T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
]

transform_3_0 = [
    MT.RandGaussianSharpen(prob=0.5, alpha=(0.0, 2.0)),


T.RandomApply([T.RandomAffine(degrees=(-30, 30))], p=0.5)
T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5)
T.RandomHorizontalFlip(p=0.5)
T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.4))
CT.RandomContrastByMultiply(p=0.5, multiplier=(0.01, 0.1))
CT.RandomSaltAndPepperNoise(p=0.5, density=(0.005, 0.01))
T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5)
T.RandomApply([T.RandomAffine(degrees=(-45, 45))], p=0.5)
CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35))
T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5)
T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.5)
T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))], p=0.5)
T.RandomApply([T.RandomResizedCrop(size=(600, 400), scale=(0.85, 1.15), ratio=(3.0 / 2.0, 3.0 / 2.0), antialias=True)], p=0.5)
CT.RandomBrightnessByAdd(p=0.5, delta=(0.25, 0.35))
T.RandomApply([T.RandomAffine(degrees=(-15, 15))], p=0.5)
T.RandomApply([T.RandomAffine(degrees=0, scale=(1.0, 1.15))], p=0.5)
T.RandomApply([T.RandomAffine(degrees=0, shear=(0, 0, -15, 15))], p=0.5)
CT.RandomNoise(p=0.5, mean=0, std=0.05)
CT.RandomNoiseWithFV(p=0.5, fv=(0.5, 1.0))
T.RandomAdjustSharpness(sharpness_factor=3.0, p=0.5)
CT.RandomNoiseWithFV(p=0.5, fv=(1.0, 1.5))
T.RandomApply([T.RandomAffine(degrees=0, scale=(0.85, 1.15))], p=0.5)
CT.RandomBrightnessByAdd(p=0.5, delta=(-0.25, 0.25))
MT.RandShiftIntensity(prob=0.5, offsets=(-0.1, 0.1))
MT.RandGaussianSharpen(prob=0.5, alpha=(0.0, 2.0))
T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5)
T.RandomAdjustSharpness(sharpness_factor=2.5, p=0.5)
MT.Rand2DElastic(prob=0.5, spacing=(45, 45), magnitude_range=(1.0, 3.0), padding_mode=zeros, spatial_size=(600, 400))
MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2))
MT.RandGaussianNoise(prob=0.5, mean=0, std=0.1)
T.RandomApply([T.RandomAffine(degrees=(-5, 5))], p=0.5)
MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(3, 3))
CT.RandomContrastByMultiply(p=0.5, multiplier=(0.5, 1.0))
T.RandomApply([T.ColorJitter(contrast=0.3)], p=0.5)
T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.5)