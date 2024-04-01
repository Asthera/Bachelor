# Articles

## Readed

- AlexNet.pdf
  - Default paper about building CNNs
  - Interesting ideas for augmenting data for testing
    -  At test time, the network makes a prediction by extracting five 224 × 224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions.
  - Augmenting data for training
    -  The training set was augmented by generating image translations and horizontal reflections [AlexNet.py].

  
- WhenToWarp.pdf
  - Comparing warping in data-space and in feature-space, data-space is better.
  - Methods for data-space used: 
    - affine transformations (translation, shearing, rotation)
    - elastic deformations.

  
- AutoAugment.pdf
  - A new data augmentation technique that uses algorithm based on RL to find the best augmentation policies [AutoAugment.txt].
  - Uses a search space of 16 operations and 10 magnitudes.
  - Operations: 
    - ShearX/Y, TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, So- larize, Posterize, Contrast, Color, Brightness, Sharpness, Cutout [12], Sample Pairing [24].2
  - Dataset: CIFAR-10, SVHN, ImageNet
  - 5 sub-policies, each with 2 operations

  
- Paired_Augmentation_for_Improved_Image_Classification_using_Neural_Network_Models.pdf
  -  Shear, Vertical Flip like single augmentations
  -  Paired augmentations(Simple + Complex) are better than single augmentations
     - Dataset 1 (Brain Tumor dataset):
       - sample pairing (an advanced  technique applied offline) with shear (a simple  technique applied online.
     - Dataset 2 (White Blood Cell dataset):
       - shear (advanced, offline) with vertical flipping (simple, online).
  - Uses augmentation in online and offline manner:
    - Offline augmentation transforms all images by increasing the size of the dataset before input to the model 
    - Online augmentation transforms the images in a batch-wise manner during model training in a feedforward manner.


- Optimizing CNN based model for thyroid nodule.pdf
  - USG images of thyroid nodules
  - Data augmentation: 
    -  RandRotation, RandomXReflection, RandomYReflection, RandomXTranslation, RandomYTranslation
  - Noise remove:
   - Compared median blur and gaussian blur
     - Gaussian blur is better
  - Used morphological operations for segmentation:
    - Erosion, Dilation
  - The morphological gradient function is performed for the enhancement of the edges
  - Good discussed metrics


- An Integration of blockchain and AI.pdf
  - Only noticed ranges for data augmentation
    - |     Syntax      | Description |             Test Text              |
      |:---------------:|:-----------:|:----------------------------------:|
      | rotation_range  |     0.2     |           rotation range           |
      | rotation_scale  |    1/255    |    ratio of image magnification    |
      | horizontal_flip |      1      |  range of horizontal translation   |
      |    fill_mode    |   nearest   |    fill the image when flipping    |
      |   width_shift   |     0.2     |  range of horizontal translation   |
      |   zoom_range    |     0.5     |  ratio of randomly zooming image   |
      |  height_shift   |     0.1     |   range of vertical translation    |
      |   shear_range   |     0.1     | range of projection transformation |


- An_Improvement_for_Medical_Image_Analysis_Using_Data_Enhancement_Techniques_in_Deep_Learning.pdf
  - From text:
    - In this work, for any input image we created other new 3 images using horizontal and vertical flip and rotation by 90 degrees (Figure 1))
  - So they used offline manner of augmentation


- Effectiveness of Data Augmentation in Image Classification.pdf
  - Compared approaches(0.855, 0.890, 0.865, 0.915, 0.900, 0.890, 0.840):
    - None
    - Traditional
      - mage that is shifted, zoomed in/out, rotated, flipped, distorted, or shaded with a hue
    - CycleGAN
    - Neural Net Augmentation + NoLoss
    - Neural Net Augmentation + Content Loss
    - Neural Net Augmentation + Style
    - Control
  
## Methods

Here will be summarized good methods for data augmentation:

- Elastic Deformations
- Affine transformations
 - Translation
 - Shearing
 - Rotation
 - Horizontal Reflection
 - Vertical Reflection
 - Zooming
 - Cropping
 - Flipping
 - Color Jittering
 - Cutout
 - Cutmix
 - Mixup
 - Scale 

- Sample Pairing

