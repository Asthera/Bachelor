import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_0 = [
    None
]

# torch transforms
transform_1_0 = [
    T.RandomErasing(p=0.5, scale=(0.001, 0.01), ratio=(0.3, 3.3), value=0, inplace=False)
]

transform_1_1 = [
    T.RandomErasing(p=0.5, scale=(0.001, 0.01), ratio=(0.3, 3.3), value="random", inplace=False)
]

transform_2_0 = [
    T.RandomErasing(p=0.7, scale=(0.001, 0.01), ratio=(0.3, 3.3), value=0, inplace=False)
]

transform_2_1 = [
    T.RandomErasing(p=0.7, scale=(0.001, 0.01), ratio=(0.3, 3.3), value="random", inplace=False)
]