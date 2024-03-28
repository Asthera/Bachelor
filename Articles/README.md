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
    - Methods for data-space used: affine transformations (translation, shearing, rotation) and elastic deformations.


    

- AutoAugment.pdf
  - A new data augmentation technique that uses algorithm based on RL to find the best augmentation policies [AutoAugment.txt].
  