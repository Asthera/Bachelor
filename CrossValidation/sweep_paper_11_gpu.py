import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT

# Shear  [−15°, 15°]
transform_1 = [
    T.RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
]

# Translate [0.15, 0.15], y-axis and x-axis
transform_2 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5),
]

# Rotate  [-25°, 25°]
transform_3 = [
    T.RandomApply([T.RandomAffine(degrees=(-25, 25))], p=0.5),
]

# Add Gaussian noise with fixed variance [0.3, 0.4, 0.5]
transform_4 = [
    T.RandomApply(T.RandomChoice([CT.RandomNoiseWithFV(p=1.0, fv=(0.3, 0.3)),
                                    CT.RandomNoiseWithFV(p=1.0, fv=(0.4, 0.4)),
                                    CT.RandomNoiseWithFV(p=1.0, fv=(0.5, 0.5))]), p=0.5)
]

# Add salt and pepper noise with density [0.01, 0.02, 0.03]
transform_5 = [
    T.RandomApply(T.RandomChoice([CT.RandomSaltAndPepperNoise(p=1.0, density=(0.01, 0.01)),
                                  CT.RandomSaltAndPepperNoise(p=1.0, density=(0.02, 0.02)),
                                  CT.RandomSaltAndPepperNoise(p=1.0, density=(0.03, 0.03))]), p=0.5)
]

# Salt and pepper noise with density [0.01, 0.02, 0.03] ans Shear  [−15°, 15°]
transform_6 = [
    T.RandomApply([T.RandomChoice([CT.RandomSaltAndPepperNoise(p=1.0, density=(0.01, 0.01)),
                                  CT.RandomSaltAndPepperNoise(p=1.0, density=(0.02, 0.02)),
                                  CT.RandomSaltAndPepperNoise(p=1.0, density=(0.03, 0.03))]),
                     T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
]

# Add Gaussian noise with fixed variance [0.3, 0.4, 0.5] and Rotate  [-25°, 25°]
transform_7 = [
    T.RandomApply([T.RandomChoice([CT.RandomNoiseWithFV(p=1.0, fv=(0.3, 0.3)),
                                    CT.RandomNoiseWithFV(p=1.0, fv=(0.4, 0.4)),
                                    CT.RandomNoiseWithFV(p=1.0, fv=(0.5, 0.5))]),
                     T.RandomAffine(degrees=(-25, 25))], p=0.5)
]

# Rotate  [-25°, 25°] and Translate [0.15, 0.15], y-axis and x-axis
transform_8 = [
    T.RandomApply([T.RandomAffine(degrees=(-25, 25)),
                     T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5)
]

# Translate [0.15, 0.15], y-axis and x-axis and Shear  [−15°, 15°]
transform_9 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15)),
                     T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
]

# Translate [0.15, 0.15] y-axis and x-axis, shear  [-15°, 15°] and Rotate  [-25°, 25°]
transform_10 = [
    T.RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15)),
                     T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15)),
                     T.RandomAffine(degrees=(-25, 25))], p=0.5)
]

#  Color shifting, sharpening, and contrast
transform_11 = [
    T.RandomApply([T.ColorJitter(brightness=(0.5, 1.5), saturation=(0.5, 1.5)),
                   T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                   T.ColorJitter(contrast=(0.5, 1.5))], p=0.5)
]

