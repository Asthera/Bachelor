import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_0 = [
    None
]

# Combinations of top 5 transforms by test-F1 score according to the folds

transform_1_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5),
    CT.RandomElasticTransform(p=0.5, alpha=(5.0, 50.0), sigma=(4.0, 5.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.01, 0.1))
]

# just mirror to previous
transform_1_1 = [
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.01, 0.1)),
    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomElasticTransform(p=0.5, alpha=(5.0, 50.0), sigma=(4.0, 5.0)),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5)
]

transform_2_0 = [
    CT.RandomBrightnessByAdd(p=1.0, delta=(0.005, 0.015)),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-30, 30, -30, 30))], p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))], p=0.5),
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2))
]

# just mirror to previous
transform_2_1 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2)),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-30, 30, -30, 30))], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5),
    CT.RandomBrightnessByAdd(p=1.0, delta=(0.005, 0.015))
]

transform_3_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.01, 0.1)),
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35))
]

# just mirror to previous
transform_3_1 = [
    CT.RandomBrightnessByAdd(p=0.5, delta=(-0.35, 0.35)),
    CT.RandomContrastByMultiply(p=0.5, multiplier=(0.01, 0.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.5),
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.3, 0.3))], p=0.5)
]

transform_4_0 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5),
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.05),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))], p=0.5),
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2))
]

# just mirror to previous
transform_4_1 = [
    MT.RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 400), rand_size=(2, 2)),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 1.0))], p=0.5),
    T.RandomHorizontalFlip(p=0.5),
    MT.RandGaussianNoise(prob=0.5, mean=0, std=0.05),
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
]

