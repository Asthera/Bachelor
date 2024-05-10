import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_0 = [
    None
]

transform_5_0_0 = [
    CT.RandomSpeckleNoise(p=0.5, mean=0, std=0.5)
]

transform_5_0_1 = [
    CT.RandomSpeckleNoise(p=0.5, mean=0, std=0.1)
]

transform_5_0_2 = [
    CT.RandomSpeckleNoise(p=0.5, mean=0, std=0.05)
]

transform_5_0_3 = [
    CT.RandomSpeckleNoise(p=0.5, mean=0, std=0.8)
]