import torchvision.transforms as T
import monai.transforms as MT
import custom.transforms as CT


transform_0 = [
    None
]

transform_1_0 = [
    MT.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.03, 0.03))
]

transform_1_1 = [
    MT.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.09, 0.09))
]

transform_1_2 = [
    MT.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.1, 0.1))
]

transform_1_3 = [
    MT.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.3, 0.3))
]

transform_1_4 = [
    MT.RandGridDistortion(num_cells=5, prob=0.5, distort_limit=(-0.5, 0.5))
]

transform_2_0 = [
    MT.RandGridDistortion(num_cells=3, prob=0.5, distort_limit=(-0.03, 0.03))
]

transform_2_1 = [
    MT.RandGridDistortion(num_cells=3, prob=0.5, distort_limit=(-0.09, 0.09))
]

transform_2_2 = [
    MT.RandGridDistortion(num_cells=3, prob=0.5, distort_limit=(-0.1, 0.1))
]

transform_2_3 = [
    MT.RandGridDistortion(num_cells=3, prob=0.5, distort_limit=(-0.3, 0.3))
]

transform_2_4 = [
    MT.RandGridDistortion(num_cells=3, prob=0.5, distort_limit=(-0.5, 0.5))
]


transform_3_0 = [
    MT.RandGridDistortion(num_cells=7, prob=0.5, distort_limit=(-0.03, 0.03))
]

transform_3_1 = [
    MT.RandGridDistortion(num_cells=7, prob=0.5, distort_limit=(-0.09, 0.09))
]

transform_3_2 = [
    MT.RandGridDistortion(num_cells=7, prob=0.5, distort_limit=(-0.1, 0.1))
]

transform_3_3 = [
    MT.RandGridDistortion(num_cells=7, prob=0.5, distort_limit=(-0.3, 0.3))
]

transform_3_4 = [
    MT.RandGridDistortion(num_cells=7, prob=0.5, distort_limit=(-0.5, 0.5))
]


transform_4_0 = [
    MT.RandGridDistortion(num_cells=9, prob=0.5, distort_limit=(-0.03, 0.03))
]

transform_4_1 = [
    MT.RandGridDistortion(num_cells=9, prob=0.5, distort_limit=(-0.09, 0.09))
]

transform_4_2 = [
    MT.RandGridDistortion(num_cells=9, prob=0.5, distort_limit=(-0.1, 0.1))
]


transform_4_3 = [
    MT.RandGridDistortion(num_cells=9, prob=0.5, distort_limit=(-0.3, 0.3))
]

transform_4_4 = [
    MT.RandGridDistortion(num_cells=9, prob=0.5, distort_limit=(-0.5, 0.5))
]

