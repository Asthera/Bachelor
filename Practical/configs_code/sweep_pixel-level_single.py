import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_0 = [
    None
]

# Noise addition
# salt and pepper noise
transform_1_0 = [
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.001, 0.001))
]

transform_1_1 = [
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.005, 0.01))
]


transform_1_2 = [
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.01, 0.01))
]

transform_1_3 = [
    CT.RandomSaltAndPepperNoise(p=0.5, density=(0.01, 0.03))
]


# Gaussian noise
transform_2_0 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.001)
]

transform_2_1 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.005)
]

transform_2_2 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.01)
]

transform_2_3 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.05)
]

transform_2_4 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.1)
]

transform_2_5 = [
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.5)
]


# Random Noise
transform_3_0 = [
    CT.RandomNoise(p=0.5, mean=0, std=0.01)
]

transform_3_1 = [
    CT.RandomNoise(p=0.5, mean=0, std=0.05)
]

transform_3_2 = [
    CT.RandomNoise(p=0.5, mean=0, std=0.1)
]

transform_3_3 = [
    CT.RandomNoise(p=0.5, mean=0, std=0.5)
]

# Noise with fv
transform_4_0 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.1, 0.1))
]

transform_4_1 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.1, 0.5))
]

transform_4_2 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.5, 0.5))
]

transform_4_3 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.5, 1.0))
]

transform_4_4 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.1, 1.0))
]

transform_4_5 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(0.5, 1.5))
]

transform_4_6 = [
    CT.RandomNoiseWithFV(p=0.5, fv=(1.0, 1.5))
]


# Speckle Noise

transform_5_0_0 = [
    CT.RandomSpeckleNoise(p=1.0, mean=0, std=0.5)
]

transform_5_0_1 = [
    CT.RandomSpeckleNoise(p=1.0, mean=0, std=0.1)
]

transform_5_0_2 = [
    CT.RandomSpeckleNoise(p=1.0, mean=0, std=0.05)
]

transform_5_0_3 = [
    CT.RandomSpeckleNoise(p=1.0, mean=0, std=0.8)
]

# Random Brightness
# By add

transform_5_0 = [
    CT.RandomBrightnessByAdd(p=1.0, delta=(0.005, 0.015))
]

transform_5_1 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(0.15, 0.25))
]

transform_5_2 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(0.25, 0.35))
]

transform_5_3 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(0.05, 0.15))
]

# with negative values
transform_5_4 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.15, 0.15))
]

transform_5_5 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.25, 0.25))
]

transform_5_6 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35))
]

# color jitter

transform_6_0 = [
    T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5)
]

transform_6_1 = [
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5)
]

transform_6_2 = [
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)], p=0.5)
]

transform_6_3 = [
    T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5)
]

transform_6_4 = [
    T.RandomApply([T.ColorJitter(brightness=0.3)], p=0.5)
]

transform_6_5 = [
    T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5)
]

transform_6_6 = [
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.5)
]


# Random Contrast
# Jitter

transform_7_0 = [
    T.RandomApply([T.ColorJitter(contrast=0.1)], p=0.5)
]

transform_7_1 = [
    T.RandomApply([T.ColorJitter(contrast=0.3)], p=0.5)
]

transform_7_2 = [
    T.RandomApply([T.ColorJitter(contrast=0.5)], p=0.5)
]


# Multiply

transform_8_0 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.5, 1.5))
]

transform_8_1 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.1, 0.5))
]

transform_8_2 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.01, 0.1))
]

transform_8_3 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.5, 1.0))
]

transform_8_4 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(1.5, 2.0))
]

transform_8_5 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.9, 1.1))
]


# Random Smooth Adjust Contrast
transform_9_0 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2))
]

transform_9_1 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(3, 3))
]

transform_9_2 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(5, 5))
]

transform_9_3 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(5, 5), gamma=(0.5, 1.5))
]

transform_9_4 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(3, 3), gamma=(0.5, 1.5))
]

transform_9_5 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2), gamma=(0.5, 1.5))
]


# Blurring

transform_10_0 = [
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5)
]

transform_10_1 = [
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)
]

transform_10_2 = [
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5)
]

transform_10_3 = [
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5))], p=0.5)
]

transform_10_4 = [
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))], p=0.5)
]

transform_10_5 = [
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))], p=0.5)
]

transform_10_6 = [
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(1.5, 2.0))], p=0.5)
]

transform_10_7 = [
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(1.5, 2.0))], p=0.5)
]

transform_10_8 = [
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.25)
]


# Sharpening
# Gaussian

transform_11_0 = [
    MT.RandGaussianSharpen(prob=0.5, alpha=(2.0, 2.0))
]

transform_11_1 = [
    MT.RandGaussianSharpen(prob=0.5, alpha=(0.0, 2.0))
]

transform_11_2 = [
    MT.RandGaussianSharpen(prob=0.5, alpha=(1.0, 5.0))
]


# Adjust Sharpness

transform_12_0 = [
    T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
]

transform_12_1 = [
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5)
]

transform_12_2 = [
    T.RandomAdjustSharpness(sharpness_factor=2.5, p=0.5)
]

transform_12_3 = [
    T.RandomAdjustSharpness(sharpness_factor=3.0, p=0.5)
]


# Random Shift Intensity

transform_13_0 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.1))
]

transform_13_1 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(0.05, 0.05))
]

transform_13_2 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(-0.1, 0.1))
]

transform_13_3 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(0.1, 0.4))
]

transform_13_4 = [
    MT.RandShiftIntensity(prob=0.5, offsets=(-0.1, 0.4))
]


