import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT

transform_0 = [
    None
]

# torch transforms
transform_1_0 = [
    CT.RandomElasticTransform(p=0.5, alpha=(5.0, 50.0), sigma=(4.0, 5.0))
]

transform_1_1 = [
    CT.RandomElasticTransform(p=0.5, alpha=(5.0, 40.0), sigma=(4.0, 5.0))
]

transform_1_2 = [
    CT.RandomElasticTransform(p=0.5, alpha=(5.0, 20.0), sigma=(3.0, 5.0))
]

transform_1_3 = [
    CT.RandomElasticTransform(p=0.5, alpha=(5.0, 10.0), sigma=(3.0, 5.0))
]

transform_1_4 = [
    CT.RandomElasticTransform(p=0.5, alpha=(20.0, 5.0), sigma=(4.0, 5.0))
]

# monai transforms
transform_2_0 = [
    MT.Rand2DElastic(prob=0.5, spacing=(30, 30), magnitude_range=(1.0, 3.0), padding_mode="zeros", spatial_size=(600, 400))
]

transform_2_1 = [
    MT.Rand2DElastic(prob=0.5, spacing=(30, 30), magnitude_range=(2.5, 3.0), padding_mode="zeros", spatial_size=(600, 400))
]

transform_2_2 = [
    MT.Rand2DElastic(prob=0.5, spacing=(30, 30), magnitude_range=(1.0, 2.0), padding_mode="zeros", spatial_size=(600, 400))
]

transform_2_3 = [
    MT.Rand2DElastic(prob=0.5, spacing=(45, 45), magnitude_range=(1.0, 3.0), padding_mode="zeros", spatial_size=(600, 400))
]