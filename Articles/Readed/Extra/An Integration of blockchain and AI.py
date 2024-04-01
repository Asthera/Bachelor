# From article, we have params for methods
# Params           Range        Description
# rotation_range    0.2     rotation range
# rotation_scale    1/255   ratio of image magnification
# horizontal_flip    1     range of horizontal translation
# fill_mode         nearest fill the image when flipping
# width_shift        0.2    range of horizontal translation
# zoom_range         0.5    ratio of randomly zooming image
# height_shift       0.1    range of vertical translation
# shear_range        0.1    range of projection transformation


# Here is inaccuracies in:
#   - rotation_range
#     it can be 0.2 of 360 degree: 72 degree
#     it can be 0.2 of 180 degree: 36 degree
#     it can be 0.2 of 90 degree: 18 degree
#     it can be 0.2 radian: 0.2 * 180 / pi = 11.46 degree

from monai.transforms import Compose, RandAffine, RandFlip, RandRotate, RandZoom, Resize





# Define the transform
transform = Compose([
    RandRotate(range=0.2, prob=1, keep_size=True),
    RandZoom(min_zoom=1, max_zoom=1.2, prob=1, keep_size=True),
    RandFlip(spatial_axis=0, prob=1),
    RandFlip(spatial_axis=1, prob=1),
    Resize(spatial_size=(224, 224)),
])