
transform: None
5
transform: RandomApply([T.RandomAffine(degrees=0, shear=(-15, 15, -15, 15))], p=0.5)
5
transform: RandomApply([T.RandomAffine(degrees=0, translate=(0.15, 0.15))], p=0.5)
5
transform: RandomApply([T.RandomAffine(degrees=(-25, 25)) ], p=0.5)
5
transform: RandomNoiseWithFV(p=0.5, fv=(0.1, 0.5))
5
transform: RandomSaltAndPepperNoise(p=0.5, density=(0.01, 0.03))
5
transform: RandGaussianNoise(prob=0.5, mean=0, std=0.002)
5
transform: RandGaussianNoise(prob=0.5, mean=0, std=0.1)
5
transform: RandomApply([MT.SavitzkyGolaySmooth(window_length=5, order=2)], p=0.5)
5
transform: RandomApply([MT.SavitzkyGolaySmooth(window_length=3, order=1)], p=0.5)
5
transform: RandomVerticalFlip(p=0.5)
5
transform: RandomHorizontalFlip(p=0.5)
5
transform: RandZoom(prob=0.5, min_zoom=1.0, max_zoom=1.2)
5
transform: RandZoom(prob=0.5, min_zoom=0.8, max_zoom=1.0)
5
transform: RandomResizedCrop(size=(600, 600), scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True)
5
transform: RandomResizedCrop(size=(600, 600), scale=(0.7, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True)
5
transform: RandomResizedCrop(size=(600, 600), scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), antialias=True)
5
transform: RandomApply([T.ColorJitter(contrast=(0.5, 1.5))], p=0.5)
5
transform: RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 600), rand_size=(5, 5))
5
transform: RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 600), rand_size=(2, 2))
5
transform: RandSmoothFieldAdjustContrast(prob=0.5, spatial_size=(600, 600), rand_size=(10, 10))
5
transform: RandomContrastByMultiply(p=0.5, multiplier=(0.75, 0.75))
5
transform: RandomApply([T.ColorJitter(brightness=(0.5, 1.5))], p=0.5)
5
transform: RandomBrightnessByAdd(p=0.5, delta=(0.15, 0.25))
5
transform: RandomBrightnessByAdd(p=0.5, delta=(0.05, 0.15))
5
transform: RandomApply([T.ColorJitter(saturation=0.5)], p=0.5)
5
transform: RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5)
5
transform: RandGaussianNoise(prob=0.5, mean=0, std=0.001)
5
transform: RandGaussianNoise(prob=0.5, mean=0, std=0.01)
5
transform: RandGaussianSharpen(prob=0.5, alpha=(1.0, 5.0))
5
transform: RandGaussianSharpen(prob=0.5)
5
transform: RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
5
transform: RandomAdjustSharpness(sharpness_factor=1.5, p=0.5)
5
transform: RandomAdjustSharpness(sharpness_factor=2.5, p=0.5)
5
transform: RandShiftIntensity(prob=0.5, offsets=(-0.1, 0.1))
5
transform: RandShiftIntensity(prob=0.5, offsets=(-0.05, 0.05))
5
transform: RandShiftIntensity(prob=0.5, offsets=(0.1, 0.1))
5
transform: RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
5
transform: RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False)